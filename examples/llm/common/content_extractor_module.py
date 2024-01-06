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

import fsspec
import mrc
import mrc.core.operators as ops
import pandas as pd
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


def read_generic_file(file):
    """
    Reads the content of a generic text file.

    Parameters
    ----------
    file : fsspec.core.OpenFile
        An instance of OpenFile representing a text file.

    Returns
    -------
    str or None
        The content of the file as a string, or None if an error occurs.
    """
    try:
        with file.fs.open(file.path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file.path}: {e}")
        return None


def read_pdf_file(file):
    """
    Reads the content of a PDF file.

    Parameters
    ----------
    file : fsspec.core.OpenFile
        An instance of OpenFile representing a PDF file.

    Returns
    -------
    bytes or None
        The content of the file as bytes, or None if an error occurs.
    """
    try:
        with file.fs.open(file.path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file.path}: {e}")
        return None


def process_content(file_path: str, content: str or bytes, is_pdf: bool, chunk_size: int, chunk_overlap: int):
    """
    Processes the content of a file (PDF or generic text) and splits it into chunks.

    Parameters
    ----------
    file_path : str
        The path to the file.
    content : str or bytes
        The content of the file.
    is_pdf : bool
        Flag to indicate if the file is a PDF.
    chunk_size : int
        Size of each chunk.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    list of dicts
        A list of dictionaries, each with a chunk of content and file metadata.
    """
    try:
        if (is_pdf):
            reader = pypdf.PdfReader(io.BytesIO(content))
            content = "".join(page.extract_text() for page in reader.pages)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       length_function=len)

        split_text = text_splitter.split_text(content)
        processed_data = []

        for chunk in split_text:
            processed_data.append({
                'title': file_path.split('/')[-1],
                'source': f"{'pdf' if is_pdf else 'txt'}:{file_path}",
                'summary': 'none',
                'content': chunk
            })

        return processed_data
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []


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

    def parse_files(open_files: typing.List[fsspec.core.OpenFile]) -> MessageMeta:
        data = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(open_files), batch_size):
                batch = open_files[i:i + batch_size]
                futures = [executor.submit(read_pdf_file if file.path.endswith('.pdf') else read_generic_file, file) for
                           file in batch]

                for file, future in zip(batch, futures):
                    content = future.result()
                    if content:
                        result = process_content(file.path, content, file.path.endswith('.pdf'), chunk_size,
                                                 chunk_overlap)
                        if result:
                            data.extend(result)

        return MessageMeta(df=pd.DataFrame(data))

    node = builder.make_node("text_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
