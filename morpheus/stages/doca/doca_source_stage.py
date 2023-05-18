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

import logging

import mrc

from morpheus.cli import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("from-doca", modes=[PipelineModes.NLP], ignore_args=["cudf_kwargs"])
class DocaSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Load messages from a file.

    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : pathlib.Path, exists = True, dir_okay = False
        Name of the file from which the messages will be read.
    iterative : boolean, default = False, is_flag = True
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages.
    repeat : int, default = 1, min = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null : bool, default = True
        Whether or not to filter rows with null 'data' column. Null values in the 'data' column can cause issues down
        the line with processing. Setting this to True is recommended.
    cudf_kwargs : dict, default = None
        keyword args passed to underlying cuDF I/O function. See the cuDF documentation for `cudf.read_csv()` and
        `cudf.read_json()` for the available options. With `file_type` == 'json', this defaults to ``{ "lines": True }``
        and with `file_type` == 'csv', this defaults to ``{}``.
    """

    def __init__(
        self,
        c: Config,
        nic_pci_address: str,
        gpu_pci_address: str,
    ):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size
        self._input_count = None
        self._max_concurrent = c.num_threads
        self._nic_pci_address = nic_pci_address
        self._gpu_pci_address = gpu_pci_address

    @property
    def name(self) -> str:
        return "from-doca"

    @property
    def input_count(self) -> int:
        """Return None for no max input count"""
        return None

    def supports_cpp_node(self):
        return True

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        if self._build_cpp_node():
            import morpheus._lib.doca as _doca
            out_stream = _doca.DocaSourceStage(builder, self.unique_name, self._nic_pci_address, self._gpu_pci_address)
        else:
            raise NotImplementedError("Does not support Python nodes")

        out_type = MessageMeta

        return out_stream, out_type
