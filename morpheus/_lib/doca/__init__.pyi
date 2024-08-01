from __future__ import annotations
import morpheus._lib.doca
import typing
import morpheus._lib.messages
import mrc.core.segment

__all__ = [
    "DocaConvertStage",
    "DocaSourceStage"
]


class DocaConvertStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str) -> None: ...
    pass
class DocaSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, nic_pci_address: str, gpu_pci_address: str, traffic_type: str) -> None: ...
    pass
