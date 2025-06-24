from __future__ import annotations
import dlp_stages._lib.regex_processor
import typing
import morpheus._lib.messages
import mrc.core.segment

__all__ = [
    "RegexProcessor"
]


class RegexProcessor(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, source_column_name: str, regex_patterns: dict[str, str], include_pattern_names: bool) -> None: ...
    pass
