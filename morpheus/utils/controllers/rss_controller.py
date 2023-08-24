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

import logging
import typing
from urllib.parse import urlparse

import feedparser

import cudf

logger = logging.getLogger(__name__)


class RSSController:

    def __init__(self, feed_input: str, batch_size: int = 128):
        """
        RSSController handles fetching and processing of RSS feed entries.

        Parameters
        ----------
        feed_input : str
            The URL or file path of the RSS feed.
        batch_size : int, optional, default = 128
            Number of feed items to accumulate before creating a DataFrame.
        """
        self._feed_input = feed_input
        self._batch_size = batch_size
        self._previous_entires = set()  # Stores the IDs of previous entries to prevent the processing of duplicates.
        # If feed_input is URL. Runs indefinitely
        self._run_indefinitely = True if RSSController.is_url(feed_input) else False

    @property
    def run_indefinitely(self):
        return self._run_indefinitely

    def parse_feed(self):
        """
        Parse the RSS feed using the feedparser library.

        Returns
        -------
        feedparser.FeedParserDict
            The parsed feed content.

        Raises
        ------
        Exception
            If the feed input is invalid or does not contain any entries.
        """
        feed = feedparser.parse(self._feed_input)

        if feed.entries:
            return feed
        else:
            raise Exception(f"Invalid feed input: {self._feed_input}. No entries found.")

    def fetch_dataframes(self) -> cudf.DataFrame:
        """
        Fetch and process RSS feed entries.

        Returns
        -------
        typing.Union[typing.List[typing.Tuple], typing.List]
            List of feed entries or None if no new entries are available.

        Raises
        ------
        Exception
            If there is error fetching or processing feed entries.
        """
        entry_accumulator = []
        current_entries = set()

        try:

            feed = self.parse_feed()

            for entry in feed.entries:
                entry_id = entry.get('id')
                current_entries.add(entry_id)

                if entry_id not in self._previous_entires:
                    entry_accumulator.append(entry)

                    if len(entry_accumulator) >= self._batch_size:
                        yield self.create_dataframe(entry_accumulator)
                        entry_accumulator.clear()

            self._previous_entires = current_entries

            # Yield any remaining entries.
            if entry_accumulator:
                df = self.create_dataframe(entry_accumulator)
                yield df

        except Exception as exc:
            raise Exception("Error fetching or processing feed entries: %s", exc)

    def create_dataframe(self, entries: typing.List[typing.Tuple]) -> cudf.DataFrame:
        """
        Create a DataFrame from accumulated entry data.

        Parameters
        ----------
        entries : typing.List[typing.Tuple]
            List of accumulated feed entries.

        Returns
        -------
        cudf.DataFrame
            A DataFrame containing feed entry data.
        """
        try:
            return cudf.DataFrame(entries)
        except Exception as exc:
            logger.error("Error creating DataFrame: %s", exc)

    @classmethod
    def is_url(cls, feed_input: str) -> bool:
        """
        Check if the provided input is a valid URL.

        Parameters
        ----------
        feed_input : str
            The input string to be checked.

        Returns
        -------
        bool
            True if the input is a valid URL, False otherwise.
        """
        try:
            parsed_url = urlparse(feed_input)
            return parsed_url.scheme != '' and parsed_url.netloc != ''
        except Exception:
            return False
