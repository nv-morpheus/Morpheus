# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import dataclasses
import logging
import queue
import threading
import typing
import warnings
from abc import abstractmethod
from functools import lru_cache
from functools import partial

import cupy as cp
import numpy as np
import tritonclient.grpc as tritonclient
from tornado.ioloop import IOLoop
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.inference.inference_stage import InferenceStage
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import ResponseMemory
from morpheus.pipeline.messages import ResponseMemoryProbs

logger = logging.getLogger(__name__)


@lru_cache(None)
def _notify_dtype_once(model_name: str, input_name: str, triton_dtype: cp.dtype, data_dtype: cp.dtype):

    can_convert = cp.can_cast(data_dtype, triton_dtype, casting="safe")

    msg = "Unexpected dtype for Triton input. "

    if (can_convert):
        msg += "Automatically converting dtype since no data loss will occur. "
    else:
        msg += "Cannot automatically convert dtype due to loss of data. "

    msg += "Model: '%s', Input Name: '%s', Expected dtype: %s, Actual dtype: %s"
    msg_args = (model_name, input_name, str(triton_dtype), str(data_dtype))

    if (can_convert):
        logger.warning(msg, *msg_args)
    else:
        raise RuntimeError(msg % msg_args)


@dataclasses.dataclass()
class TritonInOut:
    """
    Data class for model input and output configuration

    Parameters
    ----------
    name : str
        Name of the input/output in the model
    bytes : int
        Total bytes
    datatype : str
        Triton string for datatype
    shape : typing.List[int]
        Shape of input/output
    mapped_name : str
        Name of the input/output in the pipeline
    offset : int
        Offset, default value is 0
    ptr : cp.cuda.MemoryPointer
        Cupy cuda memory pointer for the input/output

    """
    name: str  # Name of the input/output in the model
    bytes: int  # Total bytes
    datatype: str  # Triton string for datatype
    shape: typing.List[int]
    mapped_name: str  # Name of the input/output in the pipeline
    offset: int = 0
    ptr: cp.cuda.MemoryPointer = None


class RecursiveQueue(queue.Queue):
    """
    Recursive queue class. Uses a `threading.RLock` instead of a `threading.Lock` for synchronization.

    Parameters
    ----------
    maxsize : int
        Max queue size. Default value is 0 which is unbounded.

    """
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)

        # Override the mutex and conditions with a recursive lock
        self.mutex = threading.RLock()

        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
        self.all_tasks_done = threading.Condition(self.mutex)


class ResourcePool:
    """
    This class provides a bounded pool of resources. Users of the pool can borrow a resource where they will
    get exclusive access to that resource until it is returned. New objects will be created if the pool is
    empty when a user requets to borrow a resource. If the max size has been hit, the user thread will be
    blocked until another thread returns a resource.

    Parameters
    ----------
    create_fn : typing.Callable[[], typing.Any]

        Function used to create new resource objects when needed.

    max_size : int, default = 10000

        Maximum number of messages in a queue.

    """
    def __init__(self, create_fn: typing.Callable[[], typing.Any], max_size: int = 1000):
        self._create_fn = create_fn
        self._max_size = max_size
        self._added_count = 0

        self._queue = RecursiveQueue()

        self._adding_condition = threading.Condition(self._queue.mutex)

        self._outstanding = []

    def _borrow(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            # Now try and create one
            with self._queue.mutex:

                # Only add it if we have room
                if (self._added_count < self._max_size):
                    self._queue.put(self._create_fn())
                    self._added_count += 1

            return self._queue.get()

    def borrow(self):
        obj = self._borrow()

        return obj

    def return_obj(self, obj):
        self._queue.put(obj)


class ShmWrapper:
    """
    This class is a wrapper around a CUDA shared memory object shared between this process and a Triton server instance.
    Since the Triton server only accepts numpy arrays as inputs, we can use this special class to pass memory references
    of inputs on the device to the server without having to go to the host eliminating serialization and network
    overhead.

    Parameters
    ----------
    client : tritonclient.InferenceServerClient
        Triton inference server client instance
    model_name : str
        Name of the model. Specifies which model can handle the inference requests that are sent to Triton
        inference server.
    config : typing.Dict[str, TritonInOut]
        Model input and output configuration. Keys represent the input/output names. Values will be a
        `TritonInOut` object

    """
    total_count = 0

    def __init__(self,
                 client: tritonclient.InferenceServerClient,
                 model_name: str,
                 config: typing.Dict[str, TritonInOut]):
        self._config = config.copy()

        self._total_bytes = 0

        for key in self._config.keys():
            self._config[key].offset = self._total_bytes
            self._total_bytes += self._config[key].bytes

        self.model_name = model_name
        self.region_name = model_name + "_{}".format(ShmWrapper.total_count)
        ShmWrapper.total_count += 1

        # Allocate the total memory
        self._memory: cp.cuda.Memory = cp.cuda.alloc(self._total_bytes).mem

        # Get memory pointers for each object
        for key in self._config.keys():
            self._config[key].ptr = cp.cuda.MemoryPointer(self._memory, self._config[key].offset)

        # Now get the registered IPC handle
        self._ipc_handle = cp.cuda.runtime.ipcGetMemHandle(self._memory.ptr)

        # Finally, regester this memory with the server. Must be base64 for some reason???
        client.register_cuda_shared_memory(self.region_name, base64.b64encode(self._ipc_handle), 0, self._total_bytes)

    def get_bytes(self, name: str):
        """
        Get the bytes needed for a particular input/output.

        Parameters
        ----------
        name : str
            Configuration name.

        Returns
        -------
        bytes
            Configuration as bytes

        """
        return self._config[name].bytes

    def get_offset(self, name: str):
        """
        Get the offset needed for a particular input/output.

        Parameters
        ----------
        name : str
            Configuration input/output name.

        Returns
        -------
        int
            Configuration offset

        """
        return self._config[name].offset

    def build_input(self, name: str, data: cp.ndarray, force_convert_inputs: bool) -> tritonclient.InferInput:
        """
        This helper function builds a Triton InferInput object that can be directly used by `tritonclient.async_infer`.
        Utilizes the config option passed in the constructor to determine the shape/size/type.

        Parameters
        ----------
        name : str
            Inference input name.
        data : cp.ndarray
            Inference input data.

        """

        expected_dtype = cp.dtype(triton_to_np_dtype(self._config[name].datatype))

        if (expected_dtype != data.dtype):

            # See if we can auto convert without loss if force_convert_inputs is False
            if (not force_convert_inputs):
                _notify_dtype_once(self.model_name, name, expected_dtype, data.dtype)

            data = data.astype(expected_dtype)

        # Create the input
        triton_input = tritonclient.InferInput(name, list(data.shape), self._config[name].datatype)

        # Set the data
        self[name].copy_from_device(data.data, data.nbytes)

        # Configure the shared memory
        # triton_input.set_shared_memory(self.region_name, data.nbytes, self.get_offset(name))

        # TODO: For EA, we will avoid shared memory to reduce risk.
        triton_input.set_data_from_numpy(data.get())

        return triton_input

    def __getitem__(self, name: str) -> cp.cuda.MemoryPointer:
        """
        Returns the `cupy.cuda.MemoryPointer` object to the internal `ShmWrapper` for the specified
        input/output name.

        :meta public:

        Parameters
        ----------
            name : str
                Input/output name.

        Returns
        -------
            cp.cuda.MemoryPointer :
                Returns the shared memory pointer for this input/output

        """
        return self._config[name].ptr


# This class is exclusively run in the worker thread. Separating the classes helps keeps the threads separate
class TritonInference:
    """
    This is a base class for all Triton inference server requests.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton
        inference server.
    server_url : str
        Triton server gRPC URL including the port.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model

    """
    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool,
                 inout_mapping: typing.Dict[str, str] = None):

        self._model_name = model_name
        self._server_url = server_url
        self._inout_mapping = inout_mapping if inout_mapping is not None else {}

        self._requires_seg_ids = False

        self._max_batch_size = c.model_max_batch_size
        self._fea_length = c.feature_length
        self._force_convert_inputs = force_convert_inputs

        # Whether or not the returned value needs a logits calc for the response
        self._needs_logits = c.mode == PipelineModes.NLP

        self._inputs: typing.Dict[str, TritonInOut] = {}
        self._outputs: typing.Dict[str, TritonInOut] = {}

        self._loop: IOLoop = None
        self._triton_client: tritonclient.InferenceServerClient = None
        self._mem_pool: ResourcePool = None

    def init(self, loop: IOLoop):
        """
        This function instantiate triton client and memory allocation for inference input and output.

        Parameters
        ----------
        loop : IOLoop
            Loop to send the response generated by future requests

        """

        self._loop = loop

        self._triton_client = tritonclient.InferenceServerClient(url=self._server_url, verbose=False)

        try:
            assert self._triton_client.is_server_live() and self._triton_client.is_server_ready(), \
                "Server is not in ready state"

            assert self._triton_client.is_model_ready(self._model_name), \
                f"Triton model {self._model_name} is not ready"

            # To make sure no shared memory regions are registered with the server.
            self._triton_client.unregister_system_shared_memory()
            self._triton_client.unregister_cuda_shared_memory()

            model_meta = self._triton_client.get_model_metadata(self._model_name, as_json=True)
            model_config = self._triton_client.get_model_config(self._model_name, as_json=True)["config"]

            # Make sure the inputs/outputs match our config
            if (int(model_meta["inputs"][0]["shape"][-1]) != self._fea_length):
                raise RuntimeError("Mismatched Sequence Length. Config specified {} but model specified {}".format(
                    self._fea_length, int(model_meta["inputs"][0]["shape"][-1])))

            # Check batch size
            if (model_config.get("max_batch_size", 0) != self._max_batch_size):

                # If the model is more, thats fine. Gen warning
                if (model_config["max_batch_size"] > self._max_batch_size):
                    warnings.warn(("Model max batch size ({}) is more than configured max batch size ({}). "
                                   "May result in sub optimal performance").format(model_config["max_batch_size"],
                                                                                   self._max_batch_size))

                # If the model is less, raise error. Cant send more to Triton than the max batch size
                if (model_config["max_batch_size"] < self._max_batch_size):
                    raise RuntimeError(
                        ("Model max batch size ({}) is less than configured max batch size ({}). "
                         "Reduce max batch size to be less than or equal to model max batch size.").format(
                             model_config["max_batch_size"], self._max_batch_size))

            shm_config = {}

            def build_inout(x: dict):
                b = np.dtype(triton_to_np_dtype(x["datatype"])).itemsize

                shape = []

                for y in x["shape"]:
                    y_int = int(y)

                    if (y_int == -1):
                        y_int = self._max_batch_size

                    shape.append(y_int)

                    b *= y_int

                mapped_name = x["name"] if x["name"] not in self._inout_mapping else self._inout_mapping[x["name"]]

                return TritonInOut(name=x["name"],
                                   bytes=b,
                                   datatype=x["datatype"],
                                   shape=shape,
                                   mapped_name=mapped_name)

            for x in model_meta["inputs"]:

                self._inputs[x["name"]] = build_inout(x)

            for x in model_meta["outputs"]:

                assert x["name"] not in self._inputs, "Input/Output names must be unique from eachother"

                self._outputs[x["name"]] = build_inout(x)

            # Combine the inputs/outputs for the shared memory
            shm_config = {**self._inputs, **self._outputs}

            def create_wrapper():
                return ShmWrapper(self._triton_client, self._model_name, shm_config)

            self._mem_pool = ResourcePool(create_fn=create_wrapper, max_size=1000)

        except InferenceServerException as ex:
            logger.exception("Exception occurred while coordinating with Triton. Exception message: \n{}\n".format(ex),
                             exc_info=ex)
            raise ex

    @abstractmethod
    def _build_response(self, result: tritonclient.InferResult) -> ResponseMemory:
        pass

    def _infer_callback(self,
                        f: asyncio.Future,
                        m: ShmWrapper,
                        result: tritonclient.InferResult,
                        error: tritonclient.InferenceServerException):

        # If its an error, return that here
        if (error is not None):
            self._loop.add_callback(f.set_exception, error)
            return

        # Build response
        response_mem = self._build_response(result)

        def tmp(mem: ResponseMemoryProbs):
            # Set result on future
            f.set_result(mem)

            # Return mempool obj
            self._mem_pool.return_obj(m)

        # We have to schedule a callback here to set the future result on the asyncio thread
        self._loop.add_callback(tmp, response_mem)

    def process(self, batch: MultiInferenceMessage, fut: asyncio.Future):
        """
        This function sends batch of events as a requests to Triton inference server using triton client API.

        Parameters
        ----------
        batch : MultiInferenceMessage
            Batch of inference messages
        fut : asyncio.Future
            Future to capture responses

        """
        mem: ShmWrapper = self._mem_pool.borrow()

        inputs: typing.List[tritonclient.InferInput] = [
            mem.build_input(input.name,
                            batch.get_input(input.mapped_name),
                            force_convert_inputs=self._force_convert_inputs) for input in self._inputs.values()
        ]

        outputs = [tritonclient.InferRequestedOutput(output.name) for output in self._outputs.values()]

        # Inference call
        self._triton_client.async_infer(model_name=self._model_name,
                                        inputs=inputs,
                                        callback=partial(self._infer_callback, fut, mem),
                                        outputs=outputs)

    def main_loop(self, loop: IOLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):
        """
        This function consumes messages (of type `MultiInferenceMessage`) posted to the internal queue and
        initiate calls to Triton inference server calls.

        Parameters
        ----------
        loop : IOLoop
            Loop to send the response generated by future requests
        inf_queue : queue.Queue
            Internal queue used as middleware to consume messages by multi threaded TritonInference
        ready_event : asyncio.Event
            ready_event

        """
        self.init(loop)

        if (ready_event is not None):
            loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

        while True:

            # Get the next work item
            message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

            batch = message[0]
            fut = message[1]

            self.process(batch, fut)


class TritonInferenceNLP(TritonInference):
    """
    This class extends TritonInference to deal with scenario-specific NLP models inference requests like building
    response.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton
        inference server.
    server_url : str
        Triton server gRPC URL including the port.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model

    """
    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool,
                 inout_mapping: typing.Dict[str, str] = None):
        # Some models use different names for the same thing. Set that here but allow user customization
        default_mapping = {
            "attention_mask": "input_mask",
        }

        default_mapping.update(inout_mapping if inout_mapping is not None else {})

        super().__init__(c,
                         model_name=model_name,
                         server_url=server_url,
                         force_convert_inputs=force_convert_inputs,
                         inout_mapping=default_mapping)

    def _build_response(self, result: tritonclient.InferResult) -> ResponseMemoryProbs:

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        if (self._needs_logits):
            output = {key: 1.0 / (1.0 + np.exp(-val)) for key, val in output.items()}

        mem = ResponseMemoryProbs(
            count=output[list(output.keys())[0]].shape[0],
            probs=cp.array(output[list(output.keys())[0]]),  # For now, only support one output
        )

        return mem


class TritonInferenceFIL(TritonInference):
    """
    This class extends `TritonInference` to deal with scenario-specific FIL models inference requests like
    building response.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton
        inference server.
    server_url : str
        Triton server gRPC URL including the port.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model

    """
    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool,
                 inout_mapping: typing.Dict[str, str] = None):
        super().__init__(c,
                         model_name=model_name,
                         server_url=server_url,
                         force_convert_inputs=force_convert_inputs,
                         inout_mapping=inout_mapping)

    def _build_response(self, result: tritonclient.InferResult) -> ResponseMemoryProbs:

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        mem = ResponseMemoryProbs(
            count=output[list(output.keys())[0]].shape[0],
            probs=cp.array(output[list(output.keys())[0]]),  # For now, only support one output
        )

        return mem


class TritonInferenceStage(InferenceStage):
    """
    This class specifies which inference implementation category (Ex: NLP/FIL) is needed for inferencing.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton inference
        server.
    server_url : str
        Triton server URL

    """
    def __init__(self, c: Config, model_name: str, server_url: str, force_convert_inputs: bool):
        super().__init__(c)

        self._model_name = model_name
        self._server_url = server_url
        self._force_convert_inputs = force_convert_inputs

        self._requires_seg_ids = False

    def _get_inference_fn(self) -> typing.Callable:

        if (Config.get().mode == PipelineModes.NLP):
            worker = TritonInferenceNLP(Config.get(),
                                        model_name=self._model_name,
                                        server_url=self._server_url,
                                        force_convert_inputs=self._force_convert_inputs)
        elif (Config.get().mode == PipelineModes.FIL):
            worker = TritonInferenceFIL(Config.get(),
                                        model_name=self._model_name,
                                        server_url=self._server_url,
                                        force_convert_inputs=self._force_convert_inputs)
        else:
            raise NotImplementedError("Unknown config mode")

        return worker.main_loop
