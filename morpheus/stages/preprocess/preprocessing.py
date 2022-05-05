# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
import logging
import typing
from abc import abstractmethod
from functools import partial

import cupy as cp
import neo
import numpy as np
import pandas as pd
import typing_utils
from neo.core import operators as ops

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import InferenceMemoryNLP
from morpheus.messages import MessageMeta
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiInferenceNLPMessage
from morpheus.messages import MultiMessage
from morpheus.pipeline.pipeline import MultiMessageStage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.cudf_subword_helper import tokenize_text_series

logger = logging.getLogger(__name__)


class DeserializeStage(MultiMessageStage):
    """
    This stage deserialize the output of `FileSourceStage`/`KafkaSourceStage` into a `MultiMessage`. This
    should be one of the first stages after the `Source` object.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._max_concurrent = c.num_threads

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "deserialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MessageMeta)

    @staticmethod
    def process_dataframe(x: MessageMeta, batch_size: int) -> typing.List[MultiMessage]:
        """
        The deserialization of the cudf is implemented in this function.

        Parameters
        ----------
        x : cudf.DataFrame
            Input rows that needs to be deserilaized.
        batch_size : int
            Batch size.

        """

        full_message = MultiMessage(meta=x, mess_offset=0, mess_count=x.count)

        # Now break it up by batches
        output = []

        for i in range(0, full_message.mess_count, batch_size):
            output.append(full_message.get_slice(i, min(i + batch_size, full_message.mess_count)))

        return output

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiMessage

        def node_fn(input: neo.Observable, output: neo.Subscriber):

            input.pipe(ops.map(partial(DeserializeStage.process_dataframe, batch_size=self._batch_size)),
                       ops.flatten()).subscribe(output)

        if CppConfig.get_should_use_cpp():
            stream = neos.DeserializeStage(seg, self.unique_name, self._batch_size)
        else:
            stream = seg.make_node_full(self.unique_name, node_fn)

        seg.make_edge(input_stream[0], stream)

        return stream, out_type


class DropNullStage(SinglePortStage):
    """
    Drop null/empty data input entries.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    column : str
        Column name to perform null check.

    """

    def __init__(self, c: Config, column: str):
        super().__init__(c)

        self._column = column

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "dropna"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (MessageMeta, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        # Finally, flatten to a single stream
        def node_fn(input: neo.Observable, output: neo.Subscriber):

            def on_next(x: MessageMeta):

                y = MessageMeta(x.df[~x.df[self._column].isna()])

                return y

            input.pipe(ops.map(on_next), ops.filter(lambda x: not x.df.empty)).subscribe(output)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(stream, node)
        stream = node

        return stream, input_stream[1]


class PreprocessBaseStage(MultiMessageStage):
    """
    This is a base pre-processing class holding general functionality for all preprocessing stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._should_log_timestamps = True

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MultiMessage, )

    @abstractmethod
    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        pass

    @abstractmethod
    def _get_preprocess_node(self, seg: neo.Segment):
        pass

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiInferenceMessage

        preprocess_fn = self._get_preprocess_fn()

        preproc_sig = inspect.signature(preprocess_fn)

        # If the innerfunction returns a type annotation, update the output type
        if (preproc_sig.return_annotation and typing_utils.issubtype(preproc_sig.return_annotation, out_type)):
            out_type = preproc_sig.return_annotation

        if CppConfig.get_should_use_cpp():
            stream = self._get_preprocess_node(seg)
        else:
            stream = seg.make_node(self.unique_name, preprocess_fn)

        seg.make_edge(input_stream[0], stream)

        return stream, out_type


class PreprocessNLPStage(PreprocessBaseStage):
    """
    NLP usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    vocab_hashfile : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
    truncation : bool
        If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly one
        output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
        than generated tokens.
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.
    add_special_tokens : bool
        Whether or not to encode the sequences with the special tokens of the BERT classification model.
    stride : int
        If `truncation` == False and the tokenized string is larger than max_length, the sequences containing the
        overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to stride
        there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be repeated on
        the second sequence and so on until the entire sentence is encoded.

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
        x : `morpheus.pipeline.messages.MultiMessage`
            Input rows received from Deserialized stage.
        vocab_hashfile : str
            Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
            using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
        do_lower_case : bool
            If set to true, original text will be lowercased before encoding.
        seq_len : int
            Limits the length of the sequence returned. If tokenized string is shorter than max_length, output will be
            padded with 0s. If the tokenized string is longer than max_length and do_truncate == False, there will be
            multiple returned sequences containing the overflowing token-ids.
        stride : int
            If do_truncate == False and the tokenized string is larger than max_length, the sequences containing the
            overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to
            stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be
            repeated on the second sequence and so on until the entire sentence is encoded.
        truncation : bool
            If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly
            one output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
            than generated tokens.
        add_special_tokens : bool
            Whether or not to encode the sequences with the special tokens of the BERT classification model.

        Returns
        -------
        `morpheus.pipeline.messages.MultiInferenceNLPMessage`
            NLP inference message.

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

        return partial(PreprocessNLPStage.pre_process_batch,
                       vocab_hash_file=self._vocab_hash_file,
                       do_lower_case=self._do_lower_case,
                       stride=self._stride,
                       seq_len=self._seq_length,
                       truncation=self._truncation,
                       add_special_tokens=self._add_special_tokens)

    def _get_preprocess_node(self, seg: neo.Segment):
        return neos.PreprocessNLPStage(seg,
                                       self.unique_name,
                                       self._vocab_hash_file,
                                       self._seq_length,
                                       self._truncation,
                                       self._do_lower_case,
                                       self._add_special_tokens,
                                       self._stride)


class PreprocessFILStage(PreprocessBaseStage):
    """
    FIL usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length
        self.features = c.fil.feature_columns

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
        x : `morpheus.pipeline.messages.MultiMessage`
            Input rows received from Deserialized stage.
        fea_len : int
            Number features are being used in the inference.
        fea_cols : typing.Tuple[str]
            List of columns that are used as features.

        Returns
        -------
        `morpheus.pipeline.messages.MultiInferenceFILMessage`
            FIL inference message.

        """

        try:
            df = x.get_meta(fea_cols)
        except KeyError:
            logger.exception("Cound not get metadat for columns.")
            return None

        # Extract just the numbers from each feature col. Not great to operate on x.meta.df here but the operations will
        # only happen once.
        for col in fea_cols:
            if (df[col].dtype == np.dtype(str) or df[col].dtype == np.dtype(object)):
                # If the column is a string, parse the number
                df[col] = df[col].str.extract(r"(\d+)", expand=False).astype("float32")
            elif (df[col].dtype != np.float32):
                # Convert to float32
                df[col] = df[col].astype("float32")

        if (isinstance(df, pd.DataFrame)):
            df = cudf.from_pandas(df)

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(df.as_gpu_matrix(order='C'))

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

    def _get_preprocess_node(self, seg: neo.Segment):
        return neos.PreprocessFILStage(seg, self.unique_name, self.features)
