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
import os
import time
from dataclasses import asdict
from dataclasses import dataclass
from urllib.parse import urlparse

import pandas as pd
import requests
import requests_cache

from morpheus.utils.verify_dependencies import _verify_deps

logger = logging.getLogger(__name__)

REQUIRED_DEPS = ('BeautifulSoup', 'feedparser')
IMPORT_ERROR_MESSAGE = "RSSController requires the bs4 and feedparser packages to be installed"

try:
    import feedparser
    from bs4 import BeautifulSoup
except ImportError:
    pass


@dataclass
class FeedStats:
    """Data class to hold error feed stats"""

    failure_count: int
    success_count: int
    last_failure: float
    last_success: float
    last_try_result: str


class RSSController:
    """
    RSSController handles fetching and processing of RSS feed entries.

    Parameters
    ----------
    feed_input : str or list[str]
        The URL or file path of the RSS feed.
    batch_size : int, optional, default = 128
        Number of feed items to accumulate before creating a DataFrame.
    run_indefinitely : bool, optional
        Whether to run the processing indefinitely. If set to True, the controller will continue fetching and processing
        If set to False, the controller will stop processing after the feed is fully fetched and processed.
        If not provided any value and if `feed_input` is of type URL, the controller will run indefinitely.
        Default is None.
    enable_cache : bool, optional, default = False
        Enable caching of RSS feed request data.
    cache_dir : str, optional, default = "./.cache/http"
        Cache directory for storing RSS feed request data.
    cooldown_interval : int, optional, default = 600
         Cooldown interval in seconds if there is a failure in fetching or parsing the feed.
    request_timeout : float, optional, default = 2.0
        Request timeout in secs to fetch the feed.
    """

    def __init__(self,
                 feed_input: str | list[str],
                 batch_size: int = 128,
                 run_indefinitely: bool = None,
                 enable_cache: bool = False,
                 cache_dir: str = "./.cache/http",
                 cooldown_interval: int = 600,
                 request_timeout: float = 2.0):
        _verify_deps(REQUIRED_DEPS, IMPORT_ERROR_MESSAGE, globals())
        if (isinstance(feed_input, str)):
            feed_input = [feed_input]

        # Convert list to set to remove any duplicate feed inputs.
        self._feed_input = set(feed_input)
        self._batch_size = batch_size
        self._previous_entries = set()  # Stores the IDs of previous entries to prevent the processing of duplicates.
        self._cooldown_interval = cooldown_interval
        self._request_timeout = request_timeout

        # Validate feed_input
        for f in self._feed_input:
            if not RSSController.is_url(f) and not os.path.exists(f):
                raise ValueError(f"Invalid URL or file path: {f}")

        if (run_indefinitely is None):
            # If feed_input is URL. Runs indefinitely
            run_indefinitely = any(RSSController.is_url(f) for f in self._feed_input)

        self._run_indefinitely = run_indefinitely

        self._session = None
        if enable_cache:
            self._session = requests_cache.CachedSession(os.path.join(cache_dir, "RSSController.sqlite"),
                                                         backend="sqlite")

        self._feed_stats_dict = {
            input:
                FeedStats(failure_count=0, success_count=0, last_failure=-1, last_success=-1, last_try_result="Unknown")
            for input in self._feed_input
        }

    @property
    def run_indefinitely(self):
        """Property that determines to run the source indefinitely"""
        return self._run_indefinitely

    @property
    def session_exist(self) -> bool:
        """Property that indicates the existence of a session."""
        return bool(self._session)

    def get_feed_stats(self, feed_url: str) -> FeedStats:
        """
        Get feed input stats.

        Parameters
        ----------
        feed_url : str
            Feed URL that is part of feed_input passed to the constructor.

        Returns
        -------
        FeedStats
            FeedStats instance for the given feed URL if it exists.

        Raises
        ------
        ValueError
            If the feed URL is not found in the feed input provided to the constructor.
        """
        if feed_url not in self._feed_stats_dict:
            raise ValueError("The feed URL is not part of the feed input provided to the constructor.")

        return self._feed_stats_dict[feed_url]

    def _get_response_text(self, url: str) -> str:
        if self.session_exist:
            response = self._session.get(url)
        else:
            response = requests.get(url, timeout=self._request_timeout)

        return response.text

    def _read_file_content(self, file_path: str) -> str:
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.read()

    def _try_parse_feed_with_beautiful_soup(self, feed_input: str, is_url: bool) -> "feedparser.FeedParserDict":

        feed_input = self._get_response_text(feed_input) if is_url else self._read_file_content(feed_input)

        soup = BeautifulSoup(feed_input, 'xml')

        # Verify whether the given feed has 'item' or 'entry' tags.
        if soup.find('item'):
            items = soup.find_all("item")
        elif soup.find('entry'):
            items = soup.find_all("entry")
        else:
            raise RuntimeError(f"Unable to find item or entry tags in {feed_input}.")

        feed_items = []
        for item in items:
            feed_item = {}
            # Iterate over each children in an item
            for child in item.children:
                if child.name is not None:
                    # If child link doesn't have a text, get it from href
                    if child.name == "link":
                        link_value = child.get_text()
                        if not link_value:
                            feed_item[child.name] = child.get('href', 'Unknown value')
                        else:
                            feed_item[child.name] = link_value
                    # To be consistant with feedparser entries, rename guid to id
                    elif child.name == "guid":
                        feed_item["id"] = child.get_text()
                    else:
                        feed_item[child.name] = child.get_text()

            feed_items.append(feed_item)

        feed = feedparser.FeedParserDict()
        feed.update({"entries": feed_items})

        return feed

    def _try_parse_feed(self, url: str) -> "feedparser.FeedParserDict":
        is_url = RSSController.is_url(url)

        fallback = False
        cache_hit = False
        is_url_with_session = is_url and self.session_exist

        if is_url_with_session:
            response = self._session.get(url)
            cache_hit = response.from_cache
            feed_input = response.text
        else:
            feed_input = url

        feed = feedparser.parse(feed_input)

        if feed["bozo"]:
            cache_hit = False

            if is_url_with_session:
                fallback = True
                logger.info("Failed to parse feed: %s. Trying to parse using feedparser directly.", url)
                feed = feedparser.parse(url)

            if feed["bozo"]:
                try:
                    logger.info("Failed to parse feed: %s, %s. Try parsing feed manually", url, feed['bozo_exception'])
                    feed = self._try_parse_feed_with_beautiful_soup(url, is_url)
                except Exception:
                    logger.error("Failed to parse the feed manually: %s", url)
                    raise

        logger.debug("Parsed feed: %s. Cache hit: %s. Fallback: %s", url, cache_hit, fallback)

        return feed

    def parse_feeds(self):
        """
        Parse the RSS feed using the feedparser library.

        Yeilds
        ------
        feedparser.FeedParserDict
            The parsed feed content.
        """
        for url in self._feed_input:
            feed_stats: FeedStats = self._feed_stats_dict[url]
            current_time = time.time()
            try:
                if ((current_time - feed_stats.last_failure) >= self._cooldown_interval):
                    feed = self._try_parse_feed(url)

                    feed_stats.last_success = current_time
                    feed_stats.success_count += 1
                    feed_stats.last_try_result = "Success"

                    yield feed

            except Exception as ex:
                logger.warning("Failed to parse feed: %s Feed stats: %s\n%s.", url, asdict(feed_stats), ex)
                feed_stats.last_failure = current_time
                feed_stats.failure_count += 1
                feed_stats.last_try_result = "Failure"

            logger.debug("Feed stats: %s", asdict(feed_stats))

    def fetch_dataframes(self):
        """
        Fetch and process RSS feed entries.

        Yeilds
        ------
        cudf.DataFrame
            A DataFrame containing feed entry data.

        Raises
        ------
        Exception
            If there is error fetching or processing feed entries.
        """
        entry_accumulator = []
        current_entries = set()

        try:

            for feed in self.parse_feeds():

                for entry in feed.entries:
                    entry_id = entry.get('id')
                    current_entries.add(entry_id)
                    if entry_id not in self._previous_entries:
                        entry_accumulator.append(entry)

                        if self._batch_size > 0 and len(entry_accumulator) >= self._batch_size:
                            yield pd.DataFrame(entry_accumulator)
                            entry_accumulator.clear()

            self._previous_entries = current_entries

            # Yield any remaining entries.
            if entry_accumulator:
                yield pd.DataFrame(entry_accumulator)
            else:
                logger.debug("No new entries found.")

        except Exception as exc:
            logger.error("Error fetching or processing feed entries: %s", exc)
            raise

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
