from __future__ import annotations
import morpheus._lib.file_types
import typing

__all__ = [
    "FileTypes",
    "determine_file_type"
]


class FileTypes():
    """
    The type of files that the `FileSourceStage` can read and `WriteToFileStage` can write. Use 'auto' to determine from the file extension.

    Members:

      Auto

      JSON

      CSV
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    Auto: morpheus._lib.file_types.FileTypes # value = <FileTypes.Auto: 0>
    CSV: morpheus._lib.file_types.FileTypes # value = <FileTypes.CSV: 2>
    JSON: morpheus._lib.file_types.FileTypes # value = <FileTypes.JSON: 1>
    __members__: dict # value = {'Auto': <FileTypes.Auto: 0>, 'JSON': <FileTypes.JSON: 1>, 'CSV': <FileTypes.CSV: 2>}
    pass
def determine_file_type(arg0: str) -> FileTypes:
    pass
