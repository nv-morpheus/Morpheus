"""
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus._lib.stages
import typing
from morpheus._lib.common import FilterSource
import morpheus._lib.common
import mrc.core.segment

__all__ = [
    "AddClassificationsStage",
    "AddScoresStage",
    "DeserializeStage",
    "FileSourceStage",
    "FilterDetectionsStage",
    "FilterSource",
    "HttpServerSourceStage",
    "InferenceClientStage",
    "KafkaSourceStage",
    "PreallocateMessageMetaStage",
    "PreallocateMultiMessageStage",
    "PreprocessFILStage",
    "PreprocessNLPStage",
    "SerializeStage",
    "WriteToFileStage"
]


class AddClassificationsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: typing.Dict[int, str], threshold: float) -> None: ...
    pass
class AddScoresStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: typing.Dict[int, str]) -> None: ...
    pass
class DeserializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, batch_size: int, ensure_sliceable_index: bool = True) -> None: ...
    pass
class FileSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, repeat: int, parser_kwargs: dict) -> None: ...
    pass
class FilterDetectionsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, threshold: float, copy: bool, filter_source: morpheus._lib.common.FilterSource, field_name: str = 'probs') -> None: ...
    pass
class HttpServerSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, bind_address: str = '127.0.0.1', port: int = 8080, endpoint: str = '/message', method: str = 'POST', accept_status: int = 201, sleep_time: float = 0.10000000149011612, queue_timeout: int = 5, max_queue_size: int = 1024, num_server_threads: int = 1, max_payload_size: int = 10485760, request_timeout: int = 30, lines: bool = False, stop_after: int = 0) -> None: ...
    pass
class InferenceClientStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, model_name: str, server_url: str, force_convert_inputs: bool, use_shared_memory: bool, needs_logits: bool, inout_mapping: typing.Dict[str, str] = {}) -> None: ...
    pass
class KafkaSourceStage(mrc.core.segment.SegmentObject):
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topic: str, batch_timeout_ms: int, config: typing.Dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True, oauth_callback: typing.Optional[function] = None) -> None: ...
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topics: typing.List[str], batch_timeout_ms: int, config: typing.Dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True, oauth_callback: typing.Optional[function] = None) -> None: ...
    pass
class PreallocateMessageMetaStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: typing.List[typing.Tuple[str, morpheus._lib.common.TypeId]]) -> None: ...
    pass
class PreallocateMultiMessageStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: typing.List[typing.Tuple[str, morpheus._lib.common.TypeId]]) -> None: ...
    pass
class PreprocessFILStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, features: typing.List[str]) -> None: ...
    pass
class PreprocessNLPStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, vocab_hash_file: str, sequence_length: int, truncation: bool, do_lower_case: bool, add_special_token: bool, stride: int, column: str) -> None: ...
    pass
class SerializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, include: typing.List[str], exclude: typing.List[str], fixed_columns: bool = True) -> None: ...
    pass
class WriteToFileStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, mode: str = 'w', file_type: morpheus._lib.common.FileTypes = FileTypes.Auto, include_index_col: bool = True, flush: bool = False) -> None: ...
    pass
__version__ = '23.11.0'
