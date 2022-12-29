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

import logging
import typing

import mrc

import morpheus._lib.stages as _stages
from morpheus._lib.type_id import TypeId
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("add-scores", rename_options={"labels": "--label"})
class AddScoresStage(SinglePortStage):
    """
    Add probability scores to each message.

    Add score labels based on probabilities calculated in inference stage. Label indexes will be looked up in
    the Config.class_labels property.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    labels : list, default = None, multiple = True, show_default = "[Config.class_labels]"
        Converts probability indexes into classification scores. Each item in the list will determine its index from the
        Config.class_labels property and must be one of the available class labels. Leave as None to add all labels in
        the Config.class_labels property.
    prefix : str, default = ""
        Prefix to add to each label. Allows adding labels different from the `Config.class_labels` property.
    probs_type : `morpheus._lib.type_id.TypeId`, default = "float32"
        Datatype of the scores columns.
    """

    def __init__(self,
                 c: Config,
                 labels: typing.List[str] = None,
                 prefix: str = "",
                 probs_type: TypeId = TypeId.FLOAT32):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._prefix = prefix
        self._class_labels = c.class_labels
        self._labels = labels if labels is not None and len(labels) > 0 else c.class_labels

        # Build the Index to Label map.
        self._idx2label = {}

        for label in self._labels:
            # All labels must be in class_labels in order to get their position
            if (label not in self._class_labels):
                logger.warning("The label '%s' is not in Config.class_labels and will be ignored", label)
                continue

            prefixed_label = self._prefix + label
            self._idx2label[self._class_labels.index(label)] = prefixed_label
            self._needed_columns[prefixed_label] = probs_type

        assert len(self._idx2label) > 0, "No labels were added to the stage"

    @property
    def name(self) -> str:
        return "add-scores"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiResponseProbsMessage`, ]
            Accepted input types.

        """
        return (MultiResponseProbsMessage, )

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def _add_labels(self, x: MultiResponseProbsMessage):

        if (x.probs.shape[1] != len(self._class_labels)):
            raise RuntimeError("Label count does not match output of model. Label count: {}, Model output: {}".format(
                len(self._class_labels), x.probs.shape[1]))

        probs_np = x.probs.get()

        for i, label in self._idx2label.items():
            x.set_meta(label, probs_np[:, i].tolist())

        # Return passthrough
        return x

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        if self._build_cpp_node():
            stream = _stages.AddScoresStage(builder, self.unique_name, len(self._class_labels), self._idx2label)
        else:
            stream = builder.make_node(self.unique_name, self._add_labels)

        builder.make_edge(input_stream[0], stream)

        # Return input unchanged
        return stream, input_stream[1]
