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
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("from-doca", modes=[PipelineModes.NLP])
class DocaSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    A source stage used to receive raw packet data from a ConnectX-6 Dx NIC.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    nic_pci_address : str
        The PCI Address of the NIC from which to recieve packets
    gpu_pci_address : str
        The PCI Address of the GPU which will receive packets
    """

    def __init__(
        self,
        c: Config,
        nic_pci_address: str,
        gpu_pci_address: str,
    ):

        super().__init__(c)

        # Attempt to import the C++ stage on creation
        try:
            # pylint: disable=c-extension-no-member
            import morpheus._lib.doca as _doca

            self._doca_source_class = _doca.DocaSourceStage
        except ImportError as ex:
            raise NotImplementedError(("The Morpheus DOCA components could not be imported. "
                                       "Ensure the DOCA components have been built and installed. Error message: ") +
                                      ex.msg) from ex

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

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def supports_cpp_node(self):
        return True

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:

        if self._build_cpp_node():
            return self._doca_source_class(builder, self.unique_name, self._nic_pci_address, self._gpu_pci_address)

        raise NotImplementedError("Does not support Python nodes")
