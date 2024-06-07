from __future__ import annotations
import morpheus._lib.doca
import typing
import datetime
import morpheus._lib.messages
import mrc.core.segment

__all__ = [
    "DocaConvertStage",
    "DocaSourceStage"
]


class DocaConvertStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_time_delta: datetime.timedelta = datetime.timedelta(seconds=3), sizes_buffer_size: int = 3145728, header_buffer_size: int = 10485760, payload_buffer_size: int = 1073741824) -> None: ...
    pass
class DocaSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, nic_pci_address: str, gpu_pci_address: str, traffic_type: str) -> None: ...
    pass
