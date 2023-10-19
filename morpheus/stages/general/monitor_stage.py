# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import mrc
from mrc.core import operators as ops
from tqdm import tqdm

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.controllers.monitor_controller import MonitorController
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.logger import LogLevels

logger = logging.getLogger(__name__)


@register_stage("monitor", ignore_args=["determine_count_fn"])
class MonitorStage(PassThruTypeMixin, SinglePortStage):
    """
    Display throughput numbers at a specific point in the pipeline.

    Monitor stage used to monitor stage performance metrics using Tqdm. Each Monitor Stage will represent one
    line in the console window showing throughput statistics. Can be set up to show an instantaneous
    throughput or average input.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    description : str, default = "Progress"
        Name to show for this Monitor Stage in the console window.
    smoothing : float
        Smoothing parameter to determine how much the throughput should be averaged. 0 = Instantaneous, 1 =
        Average.
    unit : str
        Units to show in the rate value.
    delayed_start : bool
        When delayed_start is enabled, the progress bar will not be shown until the first message is received.
        Otherwise, the progress bar is shown on pipeline startup and will begin timing immediately. In large pipelines,
        this option may be desired to give a more accurate timing.
    determine_count_fn : typing.Callable[[typing.Any], int]
        Custom function for determining the count in a message. Gets called for each message. Allows for
        correct counting of batched and sliced messages.
    log_level : `morpheus.utils.logger.LogLevels`, default = 'INFO'
        Enable this stage when the configured log level is at `log_level` or lower.
    """

    def __init__(self,
                 c: Config,
                 description: str = "Progress",
                 smoothing: float = 0.05,
                 unit: str = "messages",
                 delayed_start: bool = False,
                 determine_count_fn: typing.Callable[[typing.Any], int] = None,
                 log_level: LogLevels = LogLevels.INFO):
        super().__init__(c)

        position = MonitorController.controller_count
        self._mc = MonitorController(position=position,
                                     description=description,
                                     smoothing=smoothing,
                                     unit=unit,
                                     delayed_start=delayed_start,
                                     determine_count_fn=determine_count_fn,
                                     log_level=log_level)
        MonitorController.controller_count += 1

    @property
    def name(self) -> str:
        return "monitor"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def on_start(self):
        """
        Starts the pipeline stage's progress bar.
        """
        if self._mc.is_enabled():
            # Set the monitor interval to 0 to use prevent using tqdms monitor
            tqdm.monitor_interval = 0

            # Start the progress bar if we dont have a delayed start
            if (not self._mc.delayed_start):
                self._mc.ensure_progress_bar()

    def stop(self):
        """
        Clean up and close the progress bar.
        """
        if (self._mc.progress is not None):
            self._mc.progress.close()

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        if not self._mc.is_enabled():
            return input_node

        # Use a component so we track progress using the upstream progress engine. This will provide more accurate
        # results
        node = builder.make_node_component(self.unique_name,
                                           ops.map(self._mc.progress_sink),
                                           ops.on_completed(self._mc.sink_on_completed))

        builder.make_edge(input_node, node)

        return node
