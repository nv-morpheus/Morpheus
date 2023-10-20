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

import mrc
import mrc.core.operators as ops
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf.errors import PdfStreamError

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("from-arxiv")
class ArxivSource(SingleOutputSource):
    """
    Source stage that downloads PDFs from arxiv and converts them to dataframes
    
    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    query : `str`, default = "large language models"
        Query to use for arxiv search.
    """

    def __init__(self, c: Config, query: str = "large language models"):

        super().__init__(c)

        self._query = query
        self._max_pages = 10000

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

        self._total_pdfs = 0
        self._total_pages = 0
        self._total_chunks = 0

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

        import arxiv

        search_results = arxiv.Search(
            query=self._query,
            max_results=50,
        )

        # TODO: Move this to a config
        dir_path = "./shared-dir/dataset/pdfs/"

        for x in search_results.results():

            full_path = os.path.join(dir_path, x._get_default_filename())

            if (not os.path.exists(full_path)):
                x.download_pdf(dir_path)
                logger.debug(f"Downloaded: {full_path}")

            yield full_path

            self._total_pdfs += 1

        logger.debug(f"Downloading complete {self._total_pdfs} pages")

    def _process_pages(self, pdf_path: str):

        for _ in range(5):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                self._total_pages += len(documents)

                logger.debug(f"Processing {len(documents)}/{self._total_pages}: {pdf_path}")

                return documents
            except PdfStreamError:
                logger.error(f"Failed to load PDF (retrying): {pdf_path}")
                documents = []

        raise RuntimeError(f"Failed to load PDF: {pdf_path}")

    def _splitting_pages(self, documents: list[Document]):

        # texts1 = self._text_splitter1.split_documents(documents)
        texts = self._text_splitter.split_documents(documents)

        self._total_chunks += len(texts)

        df = pd.json_normalize([x.dict() for x in texts])

        # Rename the columns to remove the metadata prefix
        map_cols = {name: name.removeprefix("metadata.") for name in df.columns if name.startswith("metadata.")}

        df.rename(columns=map_cols, inplace=True)

        return MessageMeta(cudf.from_pandas(df))
