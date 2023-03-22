# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import typing
from abc import abstractmethod
from functools import partial
from functools import reduce

import cupy as cp
import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


class InferenceWorker:
    """
    Base class for providing implementation details for an inference stage. Create inference worker by
    subclassing this and implementing the required abstract methods. Inference stage class can then be
    assigned this worker by implementing _get_inference_worker to return your subclass.

    Parameters
    ----------
    inf_queue : `morpheus.utils.producer_consumer_queue.ProducerConsumerQueue`
        Inference queue.

    """

    def __init__(self, inf_queue: ProducerConsumerQueue) -> None:
        self._inf_queue = inf_queue

    def init(self):
        """
        By overriding this function, the resources necessary for the inference can be initiated.
        Each inference worker calls this function once.
        """

        # Nothing required in base
        pass

    def stop(self):
        """
        Override this function to stop the inference workers or carry out any additional cleanups.
        """

        pass

    def build_output_message(self, x: MultiInferenceMessage) -> MultiResponseMessage:
        """
        Create initial inference response message with result values initialized to zero. Results will be
        set in message as each inference mini-batch is processed.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiInferenceMessage`
            Batch of inference messages.

        Returns
        -------
        `morpheus.pipeline.messages.MultiResponseMessage`
            Response message with probabilities calculated from inference results.
        """

        dims = self.calc_output_dims(x)
        output_dims = (x.mess_count, *dims[1:])

        memory = TensorMemory(count=output_dims[0], tensors={'probs': cp.zeros(output_dims)})

        output_message = MultiResponseMessage.from_message(x, memory=memory)

        return output_message

    @abstractmethod
    def calc_output_dims(self, x: MultiInferenceMessage) -> typing.Tuple:
        """
        Calculates the dimensions of the inference output message data given an input message.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiInferenceMessage`
            Pipeline inference input batch before splitting into smaller inference batches.

        Returns
        -------
        typing.Tuple
            Output dimensions of response.
        """
        pass

    @abstractmethod
    def process(self, batch: MultiInferenceMessage, cb: typing.Callable[[TensorMemory], None]):
        """
        Main inference processing function. This function will be called once for each mini-batch. Once the inference is
        complete, the `cb` parameter should be used to set the response value. The callback can be called
        asynchronously.

        Parameters
        ----------
        batch : `morpheus.pipeline.messages.MultiInferenceMessage`
            Mini-batch of inference messages.
        cb : typing.Callable[[`morpheus.pipeline.messages.TensorMemory`], None]
            Callback to set the values for the inference response.

        """
        pass


class InferenceStage(MultiMessageStage):
    """
    This class serves as the base for any inference stage. Inference stages operate differently than other
    stages due to the fact that they operate in a separate thread and have their own batch size which is
    separate from the pipeline batch size. Processing the inference work in a separate thread is necessary to
    support inference types that may require exclusive use of a single thread (i.e., TensorRT) without blocking
    the main thread.

    Changing batch sizes for the inference stage requires breaking messages into smaller slices, running
    inference on the smaller slices, then recombining the inference output into the original batch size. This
    inference base class handles breaking and recombining batches and queing the inference work to be
    processed on another thread.

    Inference stages that derive from this class need to implement the `_get_inference_worker` method which
    returns an implementation of the `InferenceWorker` class. Your `InferenceWorker` class must implement the
    `process` and `calc_output_dims` methods. The `process` methods is where you provide implementation details
    on how to perform inference with the `MultiInferenceMessage` batch. The worker uses the `calc_output_dims` to
    calculate the output dimensions of the pipeline batch that inference batch results are appended to.

    To add a C++ implementation for processing inference requests, you must implement the `_get_cpp_inference_node`
    method and implement `supports_cpp_node` in your worker to return True. Your pipeline can then use your C++
    implementation by setting `use_cpp` to True in your pipeline configuration. See developer documentation for
    more details.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length

        self._thread_count = c.num_threads
        self._workers: typing.List[InferenceWorker] = []
        self._inf_queue = ProducerConsumerQueue()

        self._max_batch_size = c.model_max_batch_size

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "inference"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types to this stage.

        Returns
        -------
        typing.Tuple
            Tuple of input types.
        """
        return (MultiInferenceMessage, )

    def supports_cpp_node(self):
        # Default to False unless derived classes override this value
        return False

    @abstractmethod
    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:
        """
        Returns the main inference worker which manages requests possibly in another thread depending on which mode the
        pipeline is currently operating in.

        :meta public:

        Parameters
        ----------
        inf_queue : `morpheus.utils.producer_consumer_queue.ProducerConsumerQueue`
            Inference request queue.

        Returns
        -------
        `InferenceWorker`
            Inference worker implementation for stage.
        """
        pass

    def _get_cpp_inference_node(self, builder: mrc.Builder) -> mrc.SegmentObject:
        raise NotImplementedError("No C++ node is available for this inference type")

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiResponseMessage

        def py_inference_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            worker = self._get_inference_worker(self._inf_queue)

            worker.init()

            outstanding_requests = 0

            def on_next(x: MultiInferenceMessage):
                nonlocal outstanding_requests

                batches = self._split_batches(x, self._max_batch_size)

                output_message = worker.build_output_message(x)

                fut_list = []

                for batch in batches:
                    outstanding_requests += 1

                    completion_future = mrc.Future()

                    def set_output_fut(resp: TensorMemory, b, batch_future: mrc.Future):
                        nonlocal outstanding_requests
                        m = self._convert_one_response(output_message, b, resp)

                        outstanding_requests -= 1

                        batch_future.set_result(m)

                    fut_list.append(completion_future)

                    worker.process(batch, partial(set_output_fut, b=batch, batch_future=completion_future))

                for f in fut_list:
                    f.result()

                return output_message

            obs.pipe(ops.map(on_next)).subscribe(sub)

            assert outstanding_requests == 0, "Not all inference requests were completed"

        if (self._build_cpp_node()):
            node = self._get_cpp_inference_node(builder)
        else:
            node = builder.make_node_full(self.unique_name, py_inference_fn)

        # Set the concurrency level to be up with the thread count
        node.launch_options.pe_count = self._thread_count
        builder.make_edge(stream, node)

        stream = node

        return stream, out_type

    def _start(self):

        return super()._start()

    def stop(self):
        """
        Stops the inference workers and closes the inference queue.
        """

        for w in self._workers:
            w.stop()

        # Now stop the _inf_queue to unblock workers
        self._inf_queue.close()

    async def join(self):
        """
        On all inference worker threads, this function applies join.
        """

        # Wait for queue to be finished. This does block but it should be fine for now
        self._inf_queue.join()

        # Join all workers
        for w in self._workers:
            await w.join()

        return await super().join()

    @staticmethod
    def _split_batches(x: MultiInferenceMessage, max_batch_size: int) -> typing.List[MultiInferenceMessage]:

        out_batches = []

        id_array = cp.concatenate([cp.array([-1]), x.get_input("seq_ids")[:, 0], cp.array([-1])])

        diff_ids = cp.where(id_array[1:] != id_array[:-1])[0]

        diff_ids = diff_ids.tolist()

        head = 0
        tail = 0

        for i in range(1, len(diff_ids)):

            poss_count = diff_ids[i] - diff_ids[head]

            if (poss_count > max_batch_size):
                out_batches.append((diff_ids[head], diff_ids[tail]))

                head = tail

            tail = i

        out_batches.append((diff_ids[head], diff_ids[tail]))

        out_resp = []

        for start, stop in out_batches:

            out_resp.append(x.get_slice(start, stop))

        assert len(out_resp) > 0

        return out_resp

    @staticmethod
    def _convert_response(x: typing.Tuple[typing.List[MultiInferenceMessage], typing.List[TensorMemory]]):

        # Convert a MultiInferenceMessage into a MultiResponseMessage
        in_message = x[0]
        out_message = x[1]

        assert len(in_message) == len(out_message)

        # Get the total output size
        total_mess_count = reduce(lambda y, z: y + z.mess_count, in_message, 0)

        # Create a message data to store the entire list
        probs = cp.zeros((total_mess_count, out_message[0].get_tensor('probs').shape[1]))

        saved_offset = in_message[0].mess_offset
        saved_count = 0

        for inf, res in zip(in_message, out_message):

            # Ensure they all share the same meta object. Otherwise this doesn't work
            # assert inf.meta is saved_meta

            # Make sure we have a continuous list
            assert inf.mess_offset == saved_offset + saved_count

            assert inf.count == res.count

            # Two scenarios:
            if (inf.mess_count == inf.count):
                # In message and out message have same count. Just use probs as is
                probs[inf.offset:inf.offset + inf.count, :] = res.get_output('probs')
            else:
                mess_ids = inf.get_tensor("seq_ids")[:, 0].get().tolist()

                # Out message has more reponses, so we have to do key based blending of probs
                for i, idx in enumerate(mess_ids):
                    probs[idx, :] = cp.maximum(probs[idx, :], res.get_output('probs')[i, :])

            saved_count += inf.mess_count

        assert saved_count == total_mess_count, "Did not set every element in output"

        memory = TensorMemory(count=total_mess_count, tensors={'probs': probs})

        return MultiResponseMessage.from_message(in_message[0], mess_count=saved_count, memory=memory)

    @staticmethod
    def _convert_one_response(output: MultiResponseMessage, inf: MultiInferenceMessage, res: TensorMemory):
        # Make sure we have a continuous list
        # assert inf.mess_offset == saved_offset + saved_count
        memory = output.memory

        probs = memory.get_tensor(output.probs_tensor_name)
        resp_probs = res.get_tensor(output.probs_tensor_name)

        seq_ids = inf.get_id_tensor()

        seq_offset = seq_ids[0, 0].item() - output.mess_offset
        seq_count = (seq_ids[-1, 0].item() + 1 - seq_offset) - output.mess_offset

        # Two scenarios:
        if (inf.mess_count == inf.count):
            assert seq_count == res.count

            # In message and out message have same count. Just use probs as is
            probs[seq_offset:seq_offset + seq_count, :] = resp_probs
        else:
            assert inf.count == res.count

            mess_ids = seq_ids[:, 0].get().tolist()

            # Out message has more reponses, so we have to do key based blending of probs
            for i, idx in enumerate(mess_ids):
                probs[idx, :] = cp.maximum(probs[idx, :], resp_probs[i, :])

        return MultiResponseMessage.from_message(inf, memory=memory, offset=inf.offset, count=inf.mess_count)
