# Copyright (c) 2024, NVIDIA CORPORATION.
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
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import fsspec
import mrc
import mrc.core.operators as ops
import pandas as pd
from haystack import Document
from haystack.nodes import DocxToTextConverter
from haystack.nodes import PDFToTextConverter
from haystack.nodes import TextConverter
from haystack.nodes.file_converter import BaseConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import ValidationError

from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from .content_extractor_schema import ContentExtractorSchema

logger = logging.getLogger(__name__)

ContentExtractorLoaderFactory = ModuleLoaderFactory("file_content_extractor",
                                                    "morpheus_examples_llm",
                                                    ContentExtractorSchema)


@dataclass
class FileMeta:
    file_path: str
    file_name: str
    file_type: str


class CsvTextConverter(BaseConverter):
    """
    Converts a CSV file column content to text documents.
    """

    outgoing_edges = 1

    def convert(self,
                file_path: Path | list[Path] | str | list[str] | list[Path | str],
                meta: typing.Optional[dict[str, typing.Any]],
                remove_numeric_tables: typing.Optional[bool] = None,
                valid_languages: typing.Optional[list[str]] = None,
                encoding: typing.Optional[str] = "UTF-8",
                id_hash_keys: typing.Optional[list[str]] = None) -> list[Document]:
        """
        Load a CSV file and convert it to Documents.

        Parameters
        ----------
        file_path:
            Path to the CSV file you want to convert.
        meta:
            Optional dictionary of metadata key-value pairs that you want to append to the returned document.
        encoding:
            Optional file encoding format, default: `UTF-8`.
        id_hash_keys:
            Generates the document ID from a custom list of strings that refer to the document's attributes.
        remove_numeric_tables: bool
            Removes numeric tables from the csv.
        valid_languages:
            Valid languages

        Returns
        -------
        list[haystack.Document]
            List of documents, 1 document per line in the CSV.
        """
        if not isinstance(file_path, list):
            file_path = [file_path]

        docs: list[Document] = []
        text_column_names = ["content"]

        if meta is not None:
            text_column_names = set(meta.get("csv", {}).get("text_column_names", text_column_names))

        for path in file_path:
            df = pd.read_csv(path, encoding=encoding)
            if len(df.columns) == 0 or (not text_column_names.issubset(set(df.columns))):
                raise ValueError("The CSV file must either include a 'content' column or have a "
                                 "columns specified in the meta configuration with key 'text_column_names'.")

            df.fillna(value="", inplace=True)
            df["content"] = df[text_column_names].apply(lambda x: ' '.join(map(str, x)), axis=1)

            docs_dicts = df.to_dict(orient="records")

            for dictionary in docs_dicts:
                if meta:
                    dictionary["meta"] = meta
                if id_hash_keys:
                    dictionary["id_hash_keys"] = id_hash_keys
                docs.append(Document.from_dict(dictionary))

        return docs


def get_file_meta(open_file: fsspec.core.OpenFile) -> FileMeta:
    """
    Extract file metadata from the given open file.

    Parameters
    ----------
    open_file: fsspec.core.OpenFile
        OpenFile object

    Returns
    -------
    FileMeta
        Returns FileMeta instance.
    """
    try:
        file_path = open_file.path
        file_name = file_path.split('/')[-1]
        split_result = file_name.lower().rsplit('.', 1)

        if len(split_result) > 1:
            _, file_type = split_result
        else:
            file_type = "none"

        return FileMeta(file_path=file_path, file_name=file_name, file_type=file_type)

    except Exception as e:
        logger.error(f"Error retrieving file metadata for {open_file.path}: {e}")
        raise


def process_content(docs: list[Document], file_meta: FileMeta, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Processes the content of a file and splits it into chunks.

    Parameters
    ----------
    docs : list[Document]
        List of documents.
    file_meta: FileMeta
        FileMeta parsed information of a file path.
    chunk_size : int
        Size of each chunk.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    list of dicts
        A list of dictionaries, each with a chunk of content and file metadata.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len)

    processed_data = []

    for document in docs:
        try:
            split_text = text_splitter.split_text(document.content)

            for chunk in split_text:
                processed_data.append({
                    'title': file_meta.file_name,
                    'source': f"{file_meta.file_type}:{file_meta.file_path}",
                    'summary': 'none',
                    'content': chunk
                })

        except Exception as e:
            logger.error(f"Error processing file {file_meta.file_path} content: {e}")
            continue

    return processed_data


@register_module("file_content_extractor", "morpheus_examples_llm")
def file_content_extractor(builder: mrc.Builder):
    """
    Extracts text from PDF and TXT files and constructs a DataFrame with the extracted content.

    This module processes a batch of files, reading their contents and extracting text data to form a DataFrame.
    It can handle both PDF and TXT files. The module uses a ThreadPoolExecutor for parallel file reading.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Notes
    -----
    The `module_config` should contain:
    - 'batch_size': int, the number of files to process in parallel.
    - 'num_threads': int, the number of threads to use for parallel file reading.
    - 'chunk_size' : int, size of each chunk of document.
    - 'chunk_overlap' : int, overlap between consecutive chunks.
    - 'converters_meta' : dict, converters configuration.

    The function reads files in parallel but processes the content serially within each batch to prevent CPU contention.

    Example `module_config`
    -----------------------
    {
        "batch_size": 32,
        "num_threads": 10
    }
    """
    module_config = builder.get_current_module_config()

    try:
        extractor_config = ContentExtractorSchema(**module_config)
    except ValidationError as e:
        # Format the error message for better readability
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid configuration for file_content_extractor: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    # Use validated configurations
    batch_size = extractor_config.batch_size
    num_threads = extractor_config.num_threads
    chunk_size = extractor_config.chunk_size
    chunk_overlap = extractor_config.chunk_overlap
    converters_meta = extractor_config.converters_meta

    converters = {
        "pdf": PDFToTextConverter(),
        "csv": CsvTextConverter(),
        "docx": DocxToTextConverter(valid_languages=["de", "en"]),
        "txt": TextConverter()
    }

    chunk_params = {
        file_type: {
            "chunk_size": converters_meta.get(file_type, {}).get("chunk_size", chunk_size),
            "chunk_overlap": converters_meta.get(file_type, {}).get("chunk_overlap", chunk_overlap)
        }
        for file_type in converters.keys()
    }

    def parse_files(open_files: typing.List[fsspec.core.OpenFile]) -> MessageMeta:
        data = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(open_files), batch_size):
                batch = open_files[i:i + batch_size]
                futures = []
                files_meta = []
                for open_file in batch:
                    try:
                        file_meta: FileMeta = get_file_meta(open_file=open_file)
                        converter = converters.get(file_meta.file_type, TextConverter())
                        futures.append(executor.submit(converter.convert, file_meta.file_path, converters_meta))
                        files_meta.append(file_meta)

                    except Exception as e:
                        logger.error(f"Error processing file {open_file.path}: {e}")

                for file_meta, future in zip(files_meta, futures):
                    docs = future.result()
                    if docs:
                        file_type_chunk_params = chunk_params[file_meta.file_type]
                        result = process_content(docs,
                                                 file_meta,
                                                 file_type_chunk_params["chunk_size"],
                                                 file_type_chunk_params["chunk_overlap"])
                        if result:
                            data.extend(result)

        df_final = pd.DataFrame(data)

        return MessageMeta(df=df_final)

    node = builder.make_node("text_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
