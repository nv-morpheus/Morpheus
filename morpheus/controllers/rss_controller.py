# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import requests
import requests_cache

import cudf

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = "RSSController requires the bs4 and feedparser packages to be installed"

try:
    import feedparser
    from bs4 import BeautifulSoup
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


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
    strip_markup : bool, optional, default = False
        When true, strip HTML & XML markup from the from the content, summary and title fields.
    """

    # Fields which may contain HTML or XML content
    MARKUP_FIELDS = (
        "content",
        "summary",
        "title",
    )

    def __init__(self,
                 feed_input: str | list[str],
                 batch_size: int = 128,
                 run_indefinitely: bool = None,
                 enable_cache: bool = False,
                 cache_dir: str = "./.cache/http",
                 cooldown_interval: int = 600,
                 request_timeout: float = 2.0,
                 strip_markup: bool = False):
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        if (isinstance(feed_input, str)):
            feed_input = [feed_input]

        # Convert list to set to remove any duplicate feed inputs.
        self._feed_input = set(feed_input)
        self._batch_size = batch_size
        self._previous_entries = set()  # Stores the IDs of previous entries to prevent the processing of duplicates.
        self._cooldown_interval = cooldown_interval
        self._request_timeout = request_timeout
        self._strip_markup = strip_markup

        # Validate feed_input
        for f in self._feed_input:
            if not RSSController.is_url(f) and not os.path.exists(f):
                raise ValueError(f"Invalid URL or file path: {f}")

        if (run_indefinitely is None):
            # If feed_input is URL. Runs indefinitely
            run_indefinitely = any(RSSController.is_url(f) for f in self._feed_input)

        self._run_indefinitely = run_indefinitely
        self._enable_cache = enable_cache

        if enable_cache:
            self._session = requests_cache.CachedSession(os.path.join(cache_dir, "RSSController.sqlite"),
                                                         backend="sqlite")
        else:
            self._session = requests.session()

        self._session.headers.update({
            "User-Agent":
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        })

        self._feed_stats_dict = {
            url:
                FeedStats(failure_count=0, success_count=0, last_failure=-1, last_success=-1, last_try_result="Unknown")
            for url in self._feed_input
        }

    @property
    def run_indefinitely(self):
        """Property that determines to run the source indefinitely"""
        return self._run_indefinitely

    def get_feed_stats(self, feed_url: str) -> FeedStats:
        """
        Get feed url stats.

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
            If the feed URL is not found in the feed url provided to the constructor.
        """
        if feed_url not in self._feed_stats_dict:
            raise ValueError("The feed URL is not part of the feed url provided to the constructor.")

        return self._feed_stats_dict[feed_url]

    def _read_file_content(self, file_path: str) -> str:
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.read()

    def _try_parse_feed_with_beautiful_soup(self, feed_input: str) -> "feedparser.FeedParserDict":

        soup = BeautifulSoup(feed_input, features='xml')

        # Verify whether the given feed has 'item' or 'entry' tags.
        if soup.find('item'):
            items = soup.find_all("item")
        elif soup.find('entry'):
            items = soup.find_all("entry")
        else:
            # Check if the current logging level is DEBUG
            if (logger.getEffectiveLevel() == logging.DEBUG):
                # If DEBUG, print feed_input in full
                err_msg = f"Unable to find item or entry tags in response from {feed_input}."
            else:
                # If not DEBUG, truncate feed_input to 256 characters
                truncated_input = (feed_input[:253] + '...') if len(feed_input) > 256 else feed_input
                err_msg = (
                    f"Unable to find item or entry tags in response from feed input (truncated, set logging to debug"
                    f" for full output): {truncated_input}.")

            raise RuntimeError(err_msg)

        feed_items = []
        for item in items:
            feed_item = {}
            # Iterate over each child in an item
            for child in item.children:
                if child.name is not None:
                    # If child link doesn't have a text, get it from href
                    if child.name == "link":
                        link_value = child.get_text()
                        if not link_value:
                            feed_item[child.name] = child.get('href', 'Unknown value')
                        else:
                            feed_item[child.name] = link_value
                    # To be consistent with feedparser entries, rename guid to id
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

        if is_url:
            response = self._session.get(url, timeout=self._request_timeout)
            feed_input = response.text
            if self._enable_cache:
                cache_hit = response.from_cache
        else:
            feed_input = url

        feed = feedparser.parse(feed_input)

        if feed["bozo"]:
            fallback = True
            try:
                if not is_url:
                    # Read file content
                    feed_input = self._read_file_content(feed_input)
                # Parse feed content with beautifulsoup
                feed = self._try_parse_feed_with_beautiful_soup(feed_input)
            except Exception:
                logger.error("Failed to parse the feed manually: %s", url)
                raise

        logger.debug("Parsed feed: %s. Cache hit: %s. Fallback: %s", url, cache_hit, fallback)

        return feed

    @staticmethod
    def _strip_markup_from_field(field: str, mime_type: str) -> str:
        if mime_type.endswith("xml"):
            parser = "xml"
        else:
            parser = "html.parser"

        try:
            soup = BeautifulSoup(field, features=parser)
            return soup.get_text()
        except Exception as ex:
            logger.error("Failed to strip tags from field: %s: %s", field, ex)
            return field

    def _strip_markup_from_fields(self, entry: "feedparser.FeedParserDict"):
        """
        Strip HTML & XML tags from the content, summary and title fields.

        Per note in feedparser documentation even if a field is advertized as plain text, it may still contain HTML
        https://feedparser.readthedocs.io/en/latest/html-sanitization.html
        """
        for field in self.MARKUP_FIELDS:
            field_value = entry.get(field)
            if field_value is not None:
                if isinstance(field_value, list):
                    for field_item in field_value:
                        mime_type = field_item.get("type", "text/plain")
                        field_item["value"] = self._strip_markup_from_field(field_item["value"], mime_type)
                        field_item["type"] = "text/plain"
                else:
                    detail_field_name = f"{field}_detail"
                    detail_field: dict = entry.get(detail_field_name, {})
                    mime_type = detail_field.get("type", "text/plain")

                    entry[field] = self._strip_markup_from_field(field_value, mime_type)
                    detail_field["type"] = "text/plain"
                    entry[detail_field_name] = detail_field

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
                        if self._strip_markup:
                            self._strip_markup_from_fields(entry)

                        entry_accumulator.append(entry)

                        if self._batch_size > 0 and len(entry_accumulator) >= self._batch_size:
                            yield cudf.DataFrame(entry_accumulator)
                            entry_accumulator.clear()

            self._previous_entries = current_entries

            # Yield any remaining entries.
            if entry_accumulator:
                yield cudf.DataFrame(entry_accumulator)
            else:
                logger.debug("No new entries found.")

        except Exception as exc:
            logger.error("Error fetching or processing feed entries: %s", exc)
            raise

    @classmethod
    def is_url(cls, feed_input: str) -> bool:
        """
        Check if the provided url is a valid URL.

        Parameters
        ----------
        feed_input : str
            The url string to be checked.

        Returns
        -------
        bool
            True if the url is a valid URL, False otherwise.
        """
        try:
            parsed_url = urlparse(feed_input)
            return parsed_url.scheme != '' and parsed_url.netloc != ''
        except Exception:
            return False
