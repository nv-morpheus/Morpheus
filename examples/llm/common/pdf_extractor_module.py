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

from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


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


def process_pdf_content(file_path: str, content: bytes):
    """
    Processes the content of a PDF file and extracts text.

    Parameters
    ----------
    file_path : str
        The path to the PDF file.
    content : bytes
        The content of the PDF file.

    Returns
    -------
    dict or None
        A dictionary with extracted data or None if an error occurs.
    """
    try:
        reader = pypdf.PdfFileReader(io.BytesIO(content))
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()

        return {
            'title': file_path.split('/')[-1],
            'source': f"pdf:{file_path}",
            'summary': 'none',
            'content': text,
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None


@register_module("pdf_extractor", "morpheus_examples_llm")
def pdf_extractor(builder: mrc.Builder):
    """
    Extracts text from PDF files and constructs a DataFrame with the extracted content.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Returns
    -------
    None

    Notes
    -----
    The `module_config` should contain:
    - 'batch_size': int, the number of files to process in parallel.

    The function reads PDF files in parallel but processes the content serially
    within each batch to prevent CPU contention.
    """
    module_config = builder.get_current_module_config()
    batch_size = module_config.get("batch_size", 32)
    num_threads = module_config.get("num_threads", 10)

    def parse_files(open_files: typing.List[fsspec.core.OpenFile]) -> MessageMeta:
        data = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(open_files), batch_size):
                batch = open_files[i:i + batch_size]
                futures = [executor.submit(read_pdf_file, file) for file in batch]

                for file, future in zip(batch, futures):
                    content = future.result()
                    if content:
                        result = process_pdf_content(file.path, content)
                        if result:
                            data.append(result)

        return MessageMeta(df=pd.DataFrame(data))

    node = builder.make_node("pdf_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
