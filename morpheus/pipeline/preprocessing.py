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

import inspect
import json
import time
import typing
from abc import abstractmethod
from functools import partial

import cupy as cp
import streamz
import typing_utils

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer

from morpheus.config import Config
from morpheus.pipeline.messages import InferenceMemoryFIL
from morpheus.pipeline.messages import InferenceMemoryNLP
from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.messages import MultiInferenceFILMessage
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import MultiInferenceNLPMessage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.pipeline import MultiMessageStage
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.cudf_subword_helper import create_tokenizer
from morpheus.utils.cudf_subword_helper import tokenize_text_series


class DeserializeStage(MultiMessageStage):
    """
    This stage deserialize the output of `FileSourceStage`/`KafkaSourceStage` into a `MultiMessage`. This
    should be one of the first stages after the `Source` object.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._use_dask = c.use_dask

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "deserialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[cudf.DataFrame, morpheus.pipeline.StreamFuture[cudf.DataFrame]]
            Accepted input types

        """
        return (cudf.DataFrame, StreamFuture[cudf.DataFrame])

    @staticmethod
    def process_dataframe(x: cudf.DataFrame):
        """
        The deserialization of the cudf is implemented in this function.

        Parameters
        ----------
        x : cudf.DataFrame
            Input rows that needs to be deserilaized.

        """
        # Convert here to pandas since this will persist after the message is done
        x_pd = x.to_pandas()

        # Now determine the list of input strings before modification
        input_json = [json.dumps(y) for y in x_pd.loc[:, x_pd.columns != 'ID'].to_dict(orient="records")]

        # Add the start_time field
        x_pd["ts_start"] = round(time.time() * 1000)

        # Try to double deserialize
        def deserialize_data(y: str):
            try:
                return str(json.loads(y))
            except:  # noqa: E722
                return y

        if ("data" in x_pd):
            x_pd["data"] = x_pd["data"].apply(deserialize_data)

        # Build the message data
        meta = MessageMeta(df=x_pd, input_json=input_json)

        return MultiMessage(meta=meta, mess_offset=0, mess_count=len(x_pd))

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiMessage

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):

            stream = stream.map(DeserializeStage.process_dataframe)
            out_type = StreamFuture[MultiMessage]
        else:
            stream = stream.async_map(DeserializeStage.process_dataframe, executor=self._pipeline.thread_pool)

        return stream, out_type


class PreprocessBaseStage(MultiMessageStage):
    """
    This is a base pre-processing class holding general functionality for all preprocessing stages.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._should_log_timestamps = True

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[morpheus.messages.MultiMessage, morpheus.pipeline.StreamFuture[morpheus.messages.MultiMessage]]
            Accepted input types

        """
        return (MultiMessage, StreamFuture[MultiMessage])

    @abstractmethod
    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        pass

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiInferenceMessage

        preprocess_fn = self._get_preprocess_fn()

        preproc_sig = inspect.signature(preprocess_fn)

        # If the innerfunction returns a type annotation, update the output type
        if (preproc_sig.return_annotation and typing_utils.issubtype(preproc_sig.return_annotation, out_type)):
            out_type = preproc_sig.return_annotation

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):
            stream = stream.map(preprocess_fn)
            out_type = StreamFuture[out_type]
        else:
            stream = stream.async_map(preprocess_fn, executor=self._pipeline.thread_pool)

        return stream, out_type


class PreprocessNLPStage(PreprocessBaseStage):
    """
    NLP usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self,
                 c: Config,
                 vocab_hash_file: str,
                 truncation: bool,
                 do_lower_case: bool,
                 add_special_tokens: bool,
                 stride: int = -1):
        super().__init__(c)

        self._seq_length = c.feature_length
        self._vocab_hash_file = vocab_hash_file

        if (stride <= 0):
            # Set the stride to 75%. Works well with powers of 2
            self._stride = self._seq_length // 2
            self._stride = self._stride + self._stride // 2
        else:
            # Use the given value
            self._stride = stride

        self._truncation = truncation
        self._do_lower_case = do_lower_case
        self._add_special_tokens = add_special_tokens

        self._tokenizer: SubwordTokenizer = None

    @property
    def name(self) -> str:
        return "preprocess-nlp"

    @staticmethod
    def pre_process_batch(x: MultiMessage,
                          vocab_hash_file: str,
                          do_lower_case: bool,
                          seq_len: int,
                          stride: int,
                          truncation: bool,
                          add_special_tokens: bool) -> MultiInferenceNLPMessage:
        """
        For NLP category usecases, this function performs pre-processing.

        Parameters
        ----------
        x : morpheus.messages.MultiMessage
            Input rows recieved from Deserialized stage.
        seq_len : int
            Limits the length of the sequence returned. If tokenized string is shorter than max_length, output will be
            padded with 0s. If the tokenized string is longer than max_length and do_truncate == False, there will be
            multiple returned sequences containing the overflowing token-ids.
        stride : int
            If do_truncate == False and the tokenized string is larger than max_length, the sequences containing the
            overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to
            stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be
            repeated on the second sequence and so on until the entire sentence is encoded.
        vocab_hash_file : str
            Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
            using the cudf.utils.hash_vocab_utils.hash_vocab function.

        Returns
        -------
        morpheus.messages.MultiInferenceNLPMessage
            infer_message

        """
        text_ser = cudf.Series(x.get_meta("data"))
        tokenized = tokenize_text_series(vocab_hash_file=vocab_hash_file,
                                         do_lower_case=do_lower_case,
                                         text_ser=text_ser,
                                         seq_len=seq_len,
                                         stride=stride,
                                         truncation=truncation,
                                         add_special_tokens=add_special_tokens)
        del text_ser

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryNLP(count=tokenized.input_ids.shape[0],
                                    input_ids=tokenized.input_ids,
                                    input_mask=tokenized.input_mask,
                                    seq_ids=tokenized.segment_ids)

        infer_message = MultiInferenceNLPMessage(meta=x.meta,
                                                 mess_offset=x.mess_offset,
                                                 mess_count=x.mess_count,
                                                 memory=memory,
                                                 offset=0,
                                                 count=memory.count)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:

        # Build the tokenizer first
        # self._tokenizer = create_tokenizer(self._vocab_hash_file, self._do_lower_case)

        return partial(PreprocessNLPStage.pre_process_batch,
                       vocab_hash_file=self._vocab_hash_file,
                       do_lower_case=self._do_lower_case,
                       stride=self._stride,
                       seq_len=self._seq_length,
                       truncation=self._truncation,
                       add_special_tokens=self._add_special_tokens)


class PreprocessFILStage(PreprocessBaseStage):
    """
    FIL usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length

        self.features = [
            "nvidia_smi_log.gpu.pci.tx_util",
            "nvidia_smi_log.gpu.pci.rx_util",
            "nvidia_smi_log.gpu.fb_memory_usage.used",
            "nvidia_smi_log.gpu.fb_memory_usage.free",
            "nvidia_smi_log.gpu.bar1_memory_usage.total",
            "nvidia_smi_log.gpu.bar1_memory_usage.used",
            "nvidia_smi_log.gpu.bar1_memory_usage.free",
            "nvidia_smi_log.gpu.utilization.gpu_util",
            "nvidia_smi_log.gpu.utilization.memory_util",
            "nvidia_smi_log.gpu.temperature.gpu_temp",
            "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold",
            "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold",
            "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold",
            "nvidia_smi_log.gpu.temperature.memory_temp",
            "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold",
            "nvidia_smi_log.gpu.power_readings.power_draw",
            "nvidia_smi_log.gpu.clocks.graphics_clock",
            "nvidia_smi_log.gpu.clocks.sm_clock",
            "nvidia_smi_log.gpu.clocks.mem_clock",
            "nvidia_smi_log.gpu.clocks.video_clock",
            "nvidia_smi_log.gpu.applications_clocks.graphics_clock",
            "nvidia_smi_log.gpu.applications_clocks.mem_clock",
            "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock",
            "nvidia_smi_log.gpu.default_applications_clocks.mem_clock",
            "nvidia_smi_log.gpu.max_clocks.graphics_clock",
            "nvidia_smi_log.gpu.max_clocks.sm_clock",
            "nvidia_smi_log.gpu.max_clocks.mem_clock",
            "nvidia_smi_log.gpu.max_clocks.video_clock",
            "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock",
        ]

        assert self._fea_length == len(self.features), \
            f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

    @property
    def name(self) -> str:
        return "preprocess-fil"

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, fea_cols: typing.List[str]) -> MultiInferenceFILMessage:
        """
        For FIL category usecases, this function performs pre-processing.

        Parameters
        ----------
        x : morpheus.messages.MultiMessage
            Input rows recieved from Deserialized stage.
        fea_len : int
            Number features are being used in the inference.
        fea_cols : typing.Tuple[str]
            List of columns that are used as features.

        Returns
        -------
        morpheus.messages.MultiInferenceFILMessage
            infer_message

        """
        # Drop some extra columns we dont need
        x.meta.df.drop(x.meta.df.columns.difference(fea_cols + ["ts_start", "ts_deserialize"]), 1, inplace=True)

        # Extract just the numbers from each feature col
        for col in fea_cols:
            x.meta.df[col] = x.meta.df[col].str.extract(r"(\d+)", expand=False).astype("float32")

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(cudf.from_pandas(x.meta.df[fea_cols]).as_gpu_matrix(order='C'))

        count = data.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryFIL(count=count, input__0=data, seq_ids=seg_ids)

        infer_message = MultiInferenceFILMessage(meta=x.meta,
                                                 mess_offset=x.mess_offset,
                                                 mess_count=x.mess_count,
                                                 memory=memory,
                                                 offset=0,
                                                 count=memory.count)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        return partial(PreprocessFILStage.pre_process_batch, fea_len=self._fea_length, fea_cols=self.features)
