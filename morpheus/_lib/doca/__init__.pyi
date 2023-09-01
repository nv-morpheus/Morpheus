from __future__ import annotations
import morpheus._lib.doca
import typing
import morpheus._lib.messages
import mrc.core.segment

__all__ = [
    "DocaSourceStage"
]


class DocaSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, nic_pci_address: str, gpu_pci_address: str) -> None: ...
    pass
