# Copyright (c) 2023, NVIDIA CORPORATION.
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
import os
import typing

import mrc
import mrc.core.operators as ops
import pandas as pd

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from langchain.schema import Document

IMPORT_ERROR_MESSAGE = (
    "ArxivSource requires additional dependencies to be installed. Install them by runnign the following command: "
    "`mamba env update -n ${CONDA_DEFAULT_ENV} --file docker/conda/environments/cuda11.8_examples.yml`")


@register_stage("from-arxiv")
class ArxivSource(PreallocatorMixin, SingleOutputSource):
    """
    Source stage that downloads PDFs from arxiv and converts them to dataframes.

    This stage requires several additional dependencies to be installed. Install them by running the following command:
    `mamba env update -n ${CONDA_DEFAULT_ENV} --file docker/conda/environments/cuda11.8_examples.yml`

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    query : `str`
        Query to use for arxiv search.
    cache_dir : `str`, optional
        Directory to store downloaded PDFs in, any PDFs already in the directory will be skipped.
        This directory, will be created if it does not already exist.
    chunk_size : `int`, optional
        The number of characters to split each PDF into. This is used to split the PDF into multiple chunks each chunk
        will be converted into a row in the resulting dataframe. This value must be larger than `chunk_overlap`.
    chunk_overlap: `int`, optional
        When splitting documents into chunks, this is the number of characters that will overlap from the previus
        chunk.
    """

    def __init__(self, c: Config, query: str, cache_dir: str = "./.cache/arvix_source_cache", chunk_size: int = 1000, chunk_overlap: int = 100):

        super().__init__(c)

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        self._query = query
        self._max_pages = 10000

        if chunk_size <= chunk_overlap:
            raise ValueError(f"chunk_size must be greater than {chunk_overlap}")

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

        self._total_pdfs = 0
        self._total_pages = 0
        self._total_chunks = 0
        self._cache_dir = cache_dir

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-arxiv"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return False

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        download_pages = builder.make_source(self.unique_name + "-download", self._generate_frames())
        process_pages = builder.make_node(self.unique_name + "-process", ops.map(self._process_pages))
        process_pages.launch_options.pe_count = 6

        builder.make_edge(download_pages, process_pages)

        splitting_pages = builder.make_node(self.unique_name + "-split", ops.map(self._splitting_pages))
        # splitting_pages.launch_options.pe_count = 4

        builder.make_edge(process_pages, splitting_pages)

        out_type = MessageMeta

        return splitting_pages, out_type

    def _generate_frames(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        try:
            import arxiv
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        search_results = arxiv.Search(
            query=self._query,
            max_results=50,
        )

        for x in search_results.results():

            full_path = os.path.join(self._cache_dir, x._get_default_filename())

            if (not os.path.exists(full_path)):
                x.download_pdf(self._cache_dir)
                logger.debug("Downloaded: %s", full_path)
            else:
                logger.debug("Using cached: %s", full_path)

            yield full_path

            self._total_pdfs += 1

        logger.debug("Downloading complete %s pages", self._total_pdfs)

    def _process_pages(self, pdf_path: str):
        try:
            from langchain.document_loaders import PyPDFLoader
            from pypdf.errors import PdfStreamError
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        for _ in range(5):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                self._total_pages += len(documents)

                logger.debug("Processing %s/%s: %s", len(documents), self._total_pages, pdf_path)

                return documents
            except PdfStreamError:
                logger.error("Failed to load PDF (retrying): %s", pdf_path)
                documents = []

        raise RuntimeError(f"Failed to load PDF: {pdf_path}")

    def _splitting_pages(self, documents: list["Document"]):
        texts = self._text_splitter.split_documents(documents)

        self._total_chunks += len(texts)

        df = pd.json_normalize([x.dict() for x in texts])

        # Rename the columns to remove the metadata prefix
        map_cols = {name: name.removeprefix("metadata.") for name in df.columns if name.startswith("metadata.")}

        df.rename(columns=map_cols, inplace=True)

        return MessageMeta(cudf.from_pandas(df))
