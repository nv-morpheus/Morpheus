from __future__ import annotations
import morpheus_pass_thru_example.pass_thru_cpp
import typing
import morpheus._lib.messages
import mrc.core.segment

__all__ = [
    "PassThruStage"
]


class PassThruStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str) -> None: ...
    pass
