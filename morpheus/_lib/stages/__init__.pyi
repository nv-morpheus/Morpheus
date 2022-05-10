"""
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        """
from __future__ import annotations
import morpheus._lib.stages
import typing
import cupy
import morpheus._lib.file_types
import morpheus._lib.messages
import neo.core.node
import neo.core.segment

__all__ = [
    "AddClassificationsStage",
    "AddScoresStage",
    "DeserializeStage",
    "FileSourceStage",
    "FilterDetectionsStage",
    "InferenceClientStage",
    "KafkaSourceStage",
    "PreprocessFILStage",
    "PreprocessNLPStage",
    "SerializeStage",
    "WriteToFileStage",
    "cupy"
]


class AddClassificationsStage(neo.core.node.SegmentObject):
    """
    This is the AddClassificationsStage docstring
    """
    def __init__(self, parent: neo.core.segment.Segment, name: str, threshold: float, num_class_labels: int, idx2label: typing.Dict[int, str]) -> None: ...
    pass
class AddScoresStage(neo.core.node.SegmentObject):
    """
    This is the AddScoresStage docstring
    """
    def __init__(self, parent: neo.core.segment.Segment, name: str, num_class_labels: int, idx2label: typing.Dict[int, str]) -> None: ...
    pass
class DeserializeStage(neo.core.node.SegmentObject):
    """
    This is the DeserializeStage docstring
    """
    def __init__(self, parent: neo.core.segment.Segment, name: str, batch_size: int) -> None: ...
    pass
class FileSourceStage(neo.core.node.SegmentObject):
    """
    This is the FileSourceStage docstring
    """
    def __init__(self, parent: neo.core.segment.Segment, name: str, filename: str, repeat: int) -> None: ...
    pass
class FilterDetectionsStage(neo.core.node.SegmentObject):
    """
    This is the FilterDetectionsStage docstring
    """
    def __init__(self, parent: neo.core.segment.Segment, name: str, threshold: float) -> None: ...
    pass
class InferenceClientStage(neo.core.node.SegmentObject):
    def __init__(self, parent: neo.core.segment.Segment, name: str, model_name: str, server_url: str, force_convert_inputs: bool, use_shared_memory: bool, needs_logits: bool, inout_mapping: typing.Dict[str, str] = {}) -> None: ...
    pass
class KafkaSourceStage(neo.core.node.SegmentObject):
    def __init__(self, parent: neo.core.segment.Segment, name: str, max_batch_size: int, topic: str, batch_timeout_ms: int, config: typing.Dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False) -> None: ...
    pass
class PreprocessFILStage(neo.core.node.SegmentObject):
    def __init__(self, parent: neo.core.segment.Segment, name: str, features: typing.List[str]) -> None: ...
    pass
class PreprocessNLPStage(neo.core.node.SegmentObject):
    def __init__(self, parent: neo.core.segment.Segment, name: str, vocab_hash_file: str, sequence_length: int, truncation: bool, do_lower_case: bool, add_special_token: bool, stride: int) -> None: ...
    pass
class SerializeStage(neo.core.node.SegmentObject):
    def __init__(self, parent: neo.core.segment.Segment, name: str, include: typing.List[str], exclude: typing.List[str], fixed_columns: bool = True) -> None: ...
    pass
class WriteToFileStage(neo.core.node.SegmentObject):
    def __init__(self, parent: neo.core.segment.Segment, name: str, filename: str, mode: str = 'w', file_type: morpheus._lib.file_types.FileTypes = 0) -> None: ...
    pass
__version__ = 'dev'
