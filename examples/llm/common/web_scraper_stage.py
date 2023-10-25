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
import requests_cache
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class WebScraperStage(SinglePortStage):

    def __init__(self, c: Config, *, chunk_size, link_column: str = "link"):
        super().__init__(c)

        self._link_column = link_column
        self._chunk_size = chunk_size

        self._cache_dir = "./.cache/llm/rss/"

        # Ensure the directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size,
                                                             chunk_overlap=self._chunk_size // 10,
                                                             length_function=len)

        self._session = requests_cache.CachedSession(os.path.join("./.cache/http", "RSSDownloadStage.sqlite"),
                                                     backend="sqlite")

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

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        node = builder.make_node(self.unique_name,
                                 ops.map(self._download_and_split),
                                 ops.filter(lambda x: x is not None))
        node.launch_options.pe_count = self._config.num_threads

        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]

    def _download_and_split(self, msg: MessageMeta) -> MessageMeta:

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
                        f"Error downloading document from URL '{url}'. Returned code: {response.status_code}. With reason: '{response.reason}'"
                    )
                    continue

                raw_html = response.text

                soup = BeautifulSoup(raw_html, "html.parser")

                text = soup.get_text(strip=True)

                # article = Article(url)
                # article.download()
                # article.parse()
                # print(article.text)
                # text = article.text

                split_text = self._text_splitter.split_text(text)

                for text in split_text:
                    r = row.copy()
                    r.update({"page_content": text})
                    final_rows.append(r)

                logger.debug(f"Processed page: '{url}'. Cache hit: {response.from_cache}")

            except ValueError as e:
                logger.error(f"Error parsing document: {e}")
                continue

        # Not using cudf to avoid error: pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values
        return MessageMeta(pd.DataFrame(final_rows))
