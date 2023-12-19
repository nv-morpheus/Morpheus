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
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("from-rss")
class RSSSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Load RSS feed items into a DataFrame.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    feed_input : list[str]
        The URL or file path of the RSS feed.
    interval_secs : float, optional, default = 600
        Interval in seconds between fetching new feed items.
    stop_after: int, default = 0
        Stops ingesting after emitting `stop_after` records (rows in the dataframe). Useful for testing. Disabled if `0`
    batch_size : int, optional, default = None
        Number of feed items to accumulate before creating a DataFrame.
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
                 c: Config,
                 feed_input: list[str],
                 interval_secs: float = 600,
                 stop_after: int = 0,
                 run_indefinitely: bool = None,
                 batch_size: int = None,
                 enable_cache: bool = False,
                 cache_dir: str = "./.cache/http",
                 cooldown_interval: int = 600,
                 request_timeout: float = 2.0):
        super().__init__(c)
        self._stop_requested = False
        self._stop_after = stop_after
        self._interval_secs = interval_secs

        if (batch_size is None):
            batch_size = c.pipeline_batch_size

        if (stop_after > 0):
            if (run_indefinitely):
                raise ValueError("Cannot set both `stop_after` and `run_indefinitely` to True.")

            run_indefinitely = False

        self._records_emitted = 0
        self._controller = RSSController(feed_input=feed_input,
                                         batch_size=batch_size,
                                         run_indefinitely=run_indefinitely,
                                         enable_cache=enable_cache,
                                         cache_dir=cache_dir,
                                         cooldown_interval=cooldown_interval,
                                         request_timeout=request_timeout)

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

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _fetch_feeds(self) -> MessageMeta:
        """
        Fetch RSS feed entries and yield as MessageMeta object.
        """

        while (not self._stop_requested):
            try:
                for df in self._controller.fetch_dataframes():
                    df_size = len(df)

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Received %d new entries...", df_size)
                        logger.debug("Emitted %d records so far.", self._records_emitted)

                    yield MessageMeta(df=df)

                    self._records_emitted += df_size

                    if (self._stop_after > 0 and self._records_emitted >= self._stop_after):
                        self._stop_requested = True
                        logger.debug("Stop limit reached... preparing to halt the source.")
                        break

            except Exception as exc:
                if not self._controller.run_indefinitely:
                    logger.error("Failed either in the process of fetching or processing entries: %d.", exc)
                    raise

            if not self._controller.run_indefinitely:
                self._stop_requested = True
                continue

            logger.debug("Waiting for %d seconds before fetching again...", self._interval_secs)
            time.sleep(self._interval_secs)

        logger.debug("Source stopped.")

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._fetch_feeds)
