# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import io
import logging
import typing
from concurrent.futures import ThreadPoolExecutor
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

from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


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
            A dictionary of metadata key-value pairs that you want to append to the returned document. It's optional.
        encoding:
            Specifies the file encoding. It's optional. The default value is `UTF-8`.
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
        if meta is None:
            text_column_name = "page_content"
        else:
            text_column_name = meta["csv"]["text_column_name"]

        for path in file_path:
            df = pd.read_csv(path, encoding=encoding)
            if len(df.columns) == 0 or (text_column_name not in df.columns):
                raise ValueError("The CSV file must either include a 'page_content' column or have a "
                                 "column specified in the meta configuraton with key 'text_column_name'.")

            df.fillna(value="", inplace=True)
            df[text_column_name] = df[text_column_name].apply(lambda x: x.strip())

            df = df.rename(columns={text_column_name: "content"})
            docs_dicts = df.to_dict(orient="records")

            for dictionary in docs_dicts:
                if meta:
                    dictionary["meta"] = meta
                if id_hash_keys:
                    dictionary["id_hash_keys"] = id_hash_keys
                docs.append(Document.from_dict(dictionary))

        return docs


def get_file_extension(file_name: str) -> str:
    """
    Extract the file extension from the given file name.

    Parameters
    ----------
    file_name: str
        The name of the file.

    Returns
    -------
    str
        The file extension.
    """
    split_result = file_name.lower().rsplit('.', 1)

    if len(split_result) > 1:
        _, file_extension = split_result
    else:
        file_extension = "txt"

    return file_extension


def process_content(docs: list[Document], file_path: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Processes the content of a file and splits it into chunks.

    Parameters
    ----------
    docs : list[Document]
        List of documents.
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
    file_name = file_path.split('/')[-1]
    file_extension = get_file_extension(file_name=file_name)

    for document in docs:
        try:
            split_text = text_splitter.split_text(document.content)

            for chunk in split_text:
                processed_data.append({
                    'title': file_name, 'source': f"{file_extension}:{file_path}", 'summary': 'none', 'content': chunk
                })

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
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

    The function reads files in parallel but processes the content serially within each batch to prevent CPU contention.

    Example `module_config`
    -----------------------
    {
        "batch_size": 32,
        "num_threads": 10
    }
    """
    module_config = builder.get_current_module_config()
    batch_size = module_config.get("batch_size", 32)
    chunk_size = module_config.get("chunk_size", 1024)  # Example default value
    chunk_overlap = module_config.get("chunk_overlap", chunk_size // 10)
    num_threads = module_config.get("num_threads", 10)
    converters_meta = module_config.get("converters_meta", {})

    converters = {
        "pdf": PDFToTextConverter(),
        "csv": CsvTextConverter(),
        "docx": DocxToTextConverter(valid_languages=["de", "en"]),
        "txt": TextConverter()
    }

    def parse_files(open_files: typing.List[fsspec.core.OpenFile]) -> MessageMeta:
        data = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(open_files), batch_size):
                batch = open_files[i:i + batch_size]

                futures = []
                for open_file in batch:
                    file_path = open_file.path
                    file_name = file_path.split('/')[-1]
                    file_extension = get_file_extension(file_name=file_name)
                    futures.append(executor.submit(converters[file_extension].convert, file_path, converters_meta))

                for file, future in zip(batch, futures):
                    docs = future.result()
                    if docs:
                        result = process_content(docs, file.path, chunk_size, chunk_overlap)
                        if result:
                            data.extend(result)

        df_final = pd.DataFrame(data)

        return MessageMeta(df=df_final)

    node = builder.make_node("text_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
