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
import requests
import requests_cache
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(f"morpheus.{__name__}")


class WebScraperStage(SinglePortStage):
    """
    Stage for scraping web based content using the HTTP GET protocol.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    chunk_size : int
        Size in which to split the scraped content.
    link_column : str, default="link"
        Column which contains the links to scrape.
    enable_cache : bool, default = False
        Enables caching for requests data.
    cache_path : str, default="./.cache/http/RSSDownloadStage.sqlite"
        The path for the response caching system's sqlite database.
    """

    def __init__(self,
                 c: Config,
                 *,
                 chunk_size: int,
                 link_column: str = "link",
                 enable_cache: bool = False,
                 cache_path: str = "./.cache/http/RSSDownloadStage.sqlite"):
        super().__init__(c)

        self._link_column = link_column
        self._chunk_size = chunk_size
        self._cache_dir = "./.cache/llm/rss/"

        # Ensure the directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size,
                                                             chunk_overlap=self._chunk_size // 10,
                                                             length_function=len)

        if enable_cache:
            self._session = requests_cache.CachedSession(cache_path, backend="sqlite")
        else:
            self._session = requests.Session()

        self._session.headers.update({
            "User-Agent":
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        })

    @property
    def name(self) -> str:
        """Returns the name of this stage."""
        return "rss-download"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        node = builder.make_node(self.unique_name,
                                 ops.map(self._download_and_split),
                                 ops.filter(lambda x: x is not None))

        node.launch_options.pe_count = self._config.num_threads

        builder.make_edge(input_node, node)

        return node

    def _download_and_split(self, msg: MessageMeta) -> MessageMeta:
        """
        Uses the HTTP GET method to download/scrape the links found in the message, splits the scraped data, and stores
        it in the output, excludes output for any links which produce an error.
        """
        if self._link_column not in msg.get_column_names():
            return None

        df = msg.df

        if isinstance(df, cudf.DataFrame):
            df: pd.DataFrame = df.to_pandas()

        # Convert the dataframe into a list of dictionaries
        df_dicts = df.to_dict(orient="records")

        final_rows: list[dict] = []

        for row in df_dicts:

            url = row[self._link_column]

            try:
                # Try to get the page content
                response = self._session.get(url)

                if (not response.ok):
                    logger.warning(
                        "Error downloading document from URL '%s'. " + "Returned code: %s. With reason: '%s'",
                        url,
                        response.status_code,
                        response.reason)
                    continue

                raw_html = response.text

                soup = BeautifulSoup(raw_html, "html.parser")

                text = soup.get_text(strip=True, separator=' ')

                split_text = self._text_splitter.split_text(text)

                for text in split_text:
                    row_cp = row.copy()
                    row_cp.update({"page_content": text})
                    final_rows.append(row_cp)

                if isinstance(response, requests_cache.models.response.CachedResponse):
                    logger.debug("Processed page: '%s'. Cache hit: %s", url, response.from_cache)
                else:
                    logger.debug("Processed page: '%s'", url)

            except ValueError as exc:
                logger.error("Error parsing document: %s", exc)
                continue
            except Exception as exc:
                logger.error("Error downloading document from URL '%s'. Error: %s", url, exc)
                continue

        # Not using cudf to avoid error: pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values
        return MessageMeta(pd.DataFrame(final_rows))
