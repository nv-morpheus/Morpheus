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

import inspect
import typing
from abc import abstractmethod

import mrc
import typing_utils
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stage_schema import StageSchema


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

        self._preprocess_fn = None
        self._should_log_timestamps = True

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MultiMessage, )

    def compute_schema(self, schema: StageSchema):
        out_type = MultiInferenceMessage

        self._preprocess_fn = self._get_preprocess_fn()
        preproc_sig = inspect.signature(self._preprocess_fn)

        # If the innerfunction returns a type annotation, update the output type
        if (preproc_sig.return_annotation
                and typing_utils.issubtype(preproc_sig.return_annotation, MultiInferenceMessage)):
            out_type = preproc_sig.return_annotation

        schema.output_schema.set_type(out_type)

    @abstractmethod
    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        pass

    @abstractmethod
    def _get_preprocess_node(self, builder: mrc.Builder) -> mrc.SegmentObject:
        pass

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        assert self._preprocess_fn is not None, "Preprocess function not set"
        if self._build_cpp_node():
            node = self._get_preprocess_node(builder)
            node.launch_options.pe_count = self._config.num_threads
        else:
            node = builder.make_node(self.unique_name, ops.map(self._preprocess_fn))

        builder.make_edge(input_node, node)

        return node
