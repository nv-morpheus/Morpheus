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
import time

import mrc

from morpheus.cli import register_stage
from morpheus.config import Config
from morpheus.controllers.rss_controller import RSSController
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("from-rss")
class RSSSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Load RSS feed items into a pandas DataFrame.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    feed_input : str
        The URL or file path of the RSS feed.
    interval_secs : float, optional, default = 600
        Interval in seconds between fetching new feed items.
    stop_after: int, default = 0
        Stops ingesting after emitting `stop_after` records (rows in the dataframe). Useful for testing. Disabled if `0`
    max_retries : int, optional, default = 3
        Maximum number of retries for fetching entries on exception.
    """

    def __init__(self,
                 c: Config,
                 feed_input: str,
                 interval_secs: float = 600,
                 stop_after: int = 0,
                 max_retries: int = 5):
        super().__init__(c)
        self._stop_requested = False
        self._stop_after = stop_after
        self._interval_secs = interval_secs
        self._max_retries = max_retries

        self._records_emitted = 0
        self._controller = RSSController(feed_input=feed_input, batch_size=c.pipeline_batch_size)

    @property
    def name(self) -> str:
        return "from-rss"

    def stop(self):
        """
        Stop the RSS source stage.
        """
        self._stop_requested = True
        return super().stop()

    def supports_cpp_node(self):
        return False

    def output_type(self) -> type:
        return MessageMeta

    def _fetch_feeds(self) -> MessageMeta:
        """
        Fetch RSS feed entries and yield as MessageMeta object.
        """
        retries = 0

        while (not self._stop_requested) and (retries < self._max_retries):
            try:
                for df in self._controller.fetch_dataframes():
                    df_size = len(df)
                    self._records_emitted += df_size

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Received %d new entries...", df_size)
                        logger.debug("Emitted %d records so far.", self._records_emitted)

                    yield MessageMeta(df=df)

                if not self._controller.run_indefinitely:
                    self._stop_requested = True
                    continue

                if (self._stop_after > 0 and self._records_emitted >= self._stop_after):
                    self._stop_requested = True
                    logger.debug("Stop limit reached...preparing to halt the source.")
                    continue

                logger.debug("Waiting for %d seconds before fetching again...", self._interval_secs)
                time.sleep(self._interval_secs)

            except Exception as exc:
                if not self._controller.run_indefinitely:
                    logger.error("The input provided is not a URL or a valid path, therefore, the maximum " +
                                 "retries are being overridden, and early exiting is triggered.")
                    raise RuntimeError(f"Failed to fetch feed entries : {exc}") from exc

                retries += 1
                logger.warning("Error fetching feed entries. Retrying (%d/%d)...", retries, self._max_retries)
                logger.debug("Waiting for 5 secs before retrying...")
                time.sleep(5)  # Wait before retrying

                if retries == self._max_retries:  # Check if retries exceeded the limit
                    logger.error("Max retries reached. Unable to fetch feed entries.")
                    raise RuntimeError(f"Failed to fetch feed entries after max retries: {exc}") from exc

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._fetch_feeds)
