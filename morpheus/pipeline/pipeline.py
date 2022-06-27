# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import asyncio
import logging
import os
import signal
import time
import typing

import networkx
import srf
from tqdm import tqdm

import cudf

from morpheus.config import Config
from morpheus.pipeline.receiver import Receiver
from morpheus.pipeline.sender import Sender
from morpheus.pipeline.source_stage import SourceStage
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stream_wrapper import StreamWrapper
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)


class Pipeline():
    """
    Class for building your pipeline. A pipeline for your use case can be constructed by first adding a
    `Source` via `set_source` then any number of downstream `Stage` classes via `add_stage`. The order stages
    are added with `add_stage` determines the order in which stage executions are carried out. You can use
    stages included within Morpheus or your own custom-built stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        self._source_count: int = None  # Maximum number of iterations for progress reporting. None = Unknown/Unlimited

        self._id_counter = 0

        self._sources: typing.Set[SourceStage] = set()
        self._stages: typing.Set[Stage] = set()

        self._exec_options = srf.Options()
        self._exec_options.topology.user_cpuset = "0-{}".format(c.num_threads - 1)
        self._exec_options.engine_factories.default_engine_type = srf.core.options.EngineType.Thread

        # Set the default channel size
        srf.Config.default_channel_size = c.edge_buffer_size

        self.batch_size = c.pipeline_batch_size

        self._graph = networkx.DiGraph()

        self._is_built = False
        self._build_complete = srf.Future()
        self._is_started = False

        self._srf_executor: srf.Executor = None
        self._srf_pipeline: srf.Pipeline = None

    @property
    def is_built(self) -> bool:
        return self._is_built

    def _add_id_col(self, x: cudf.DataFrame):

        # Data in stream is cudf Dataframes at this point. We need an ID column before continuing
        x.insert(0, 'ID', range(self._id_counter, self._id_counter + len(x)))
        self._id_counter += len(x)

        return x

    def add_node(self, node: StreamWrapper):

        assert node._pipeline is None or node._pipeline is self, "A stage can only be added to one pipeline at a time"

        # Add to list of stages if its a stage, not a source
        if (isinstance(node, Stage)):
            self._stages.add(node)
        elif (isinstance(node, SourceStage)):
            self._sources.add(node)
        else:
            raise NotImplementedError("add_node() failed. Unknown node type: {}".format(type(node)))

        node._pipeline = self

        self._graph.add_node(node)

    def add_edge(self, start: typing.Union[StreamWrapper, Sender], end: typing.Union[Stage, Receiver]):

        if (isinstance(start, StreamWrapper)):
            start_port = start.output_ports[0]
        elif (isinstance(start, Sender)):
            start_port = start

        if (isinstance(end, Stage)):
            end_port = end.input_ports[0]
        elif (isinstance(end, Receiver)):
            end_port = end

        # Ensure both are added to this pipeline
        self.add_node(start_port.parent)
        self.add_node(end_port.parent)

        start_port._output_receivers.append(end_port)
        end_port._input_senders.append(start_port)

        self._graph.add_edge(start_port.parent,
                             end_port.parent,
                             start_port_idx=start_port.port_number,
                             end_port_idx=end_port.port_number)

    def build(self):
        """
        This function sequentially activates all of the Morpheus pipeline stages passed by the users to execute a
        pipeline. For the `Source` and all added `Stage` objects, `StreamWrapper.build` will be called sequentially to
        construct the pipeline.

        Once the pipeline has been constructed, this will start the pipeline by calling `Source.start` on the source
        object.
        """
        assert not self._is_built, "Pipeline can only be built once!"
        assert len(self._sources) > 0, "Pipeline must have a source stage"

        logger.info("====Registering Pipeline====")

        self._srf_executor = srf.Executor(self._exec_options)

        self._srf_pipeline = srf.Pipeline()

        def inner_build(builder: srf.Builder):
            logger.info("====Building Pipeline====")

            # Get the list of stages and source
            source_and_stages: typing.List[StreamWrapper] = list(self._sources) + list(self._stages)

            # Now loop over stages
            for s in source_and_stages:

                if (s.can_build()):
                    s.build(builder)

            if (not all([x.is_built for x in source_and_stages])):
                # raise NotImplementedError("Circular pipelines are not yet supported!")
                logger.warning("Circular pipeline detected! Building with reduced constraints")

                for s in source_and_stages:

                    if (s.can_build(check_ports=True)):
                        s.build()

            if (not all([x.is_built for x in source_and_stages])):
                raise RuntimeError("Could not build pipeline. Ensure all types can be determined")

            # Finally, execute the link phase (only necessary for circular pipelines)
            for s in source_and_stages:

                for p in s.input_ports:
                    p.link()

            logger.info("====Building Pipeline Complete!====")
            self._build_complete.set_result(True)

            # Finally call _on_start
            self._on_start()

        self._srf_pipeline.make_segment("main", inner_build)

        self._srf_executor.register_pipeline(self._srf_pipeline)

        self._is_built = True

        logger.info("====Registering Pipeline Complete!====")

    def start(self):
        assert self._is_built, "Pipeline must be built before starting"

        logger.info("====Starting Pipeline====")

        self._srf_executor.start()

        logger.info("====Pipeline Started====")

    def stop(self):

        logger.info("====Stopping Pipeline====")
        for s in list(self._sources) + list(self._stages):
            s.stop()

        self._srf_executor.stop()

        logger.info("====Pipeline Stopped====")

    async def join(self):

        await self._srf_executor.join_async()

        # First wait for all sources to stop. This only occurs after all messages have been processed fully
        for s in list(self._sources):
            await s.join()

        # Now that there is no more data, call stop on all stages to ensure shutdown (i.e., for stages that have their
        # own worker loop thread)
        for s in list(self._stages):
            s.stop()

        # Now call join on all stages
        for s in list(self._stages):
            await s.join()

    def build_and_start(self):

        if (not self.is_built):
            try:
                self.build()
            except Exception:
                logger.exception("Error occurred during Pipeline.build(). Exiting.", exc_info=True)
                return

        self.start()

    def _on_start(self):

        # Only execute this once
        if (self._is_started):
            return

        # Stop from running this twice
        self._is_started = True

        logger.debug("Starting! Time: {}".format(time.time()))

        # Loop over all stages and call on_start if it exists
        for s in self._stages:
            s.on_start()

    def visualize(self, filename: str = None, **graph_kwargs):

        # Mimic the streamz visualization
        # 1. Create graph (already done since we use networkx under the hood)
        # 2. Readable graph
        # 3. To graphviz
        # 4. Draw
        import graphviz

        # Default graph attributes
        graph_attr = {
            "nodesep": "1",
            "ranksep": "1",
            "pad": "0.5",
        }

        # Allow user to overwrite defaults
        graph_attr.update(graph_kwargs)

        gv_graph = graphviz.Digraph(graph_attr=graph_attr)

        # Need a little different functionality for left/right vs vertical
        is_lr = graph_kwargs.get("rankdir", None) == "LR"

        start_def_port = ":e" if is_lr else ":s"
        end_def_port = ":w" if is_lr else ":n"

        def has_ports(n: StreamWrapper, is_input):
            if (is_input):
                return len(n.input_ports) > 0
            else:
                return len(n.output_ports) > 0

        if (not self.is_built):
            try:
                self.build()
            except Exception:
                logger.exception("Error occurred during Pipeline.build(). Exiting.", exc_info=True)
                return

        # The pipeline build is performed asynchronously, block until the build is completed
        self._build_complete.result()

        # Now build up the nodes
        for n, attrs in typing.cast(typing.Mapping[StreamWrapper, dict], self._graph.nodes).items():
            node_attrs = attrs.copy()

            label = ""

            show_in_ports = has_ports(n, is_input=True)
            show_out_ports = has_ports(n, is_input=False)

            # Build the ports for the node. Only show ports if there are any (Would like to have this not show for one
            # port, but the lines get all messed up)
            if (show_in_ports):
                in_port_label = " {{ {} }} | ".format(" | ".join(
                    [f"<u{x.port_number}> {x.port_number}" for x in n.input_ports]))
                label += in_port_label

            label += n.unique_name

            if (show_out_ports):
                out_port_label = " | {{ {} }}".format(" | ".join(
                    [f"<d{x.port_number}> {x.port_number}" for x in n.output_ports]))
                label += out_port_label

            if (show_in_ports or show_out_ports):
                label = f"{{ {label} }}"

            node_attrs.update({
                "label": label,
                "shape": "record",
                "fillcolor": "white",
            })
            # TODO: Eventually allow nodes to have different attributes based on type
            # node_attrs.update(n.get_graphviz_attrs())
            gv_graph.node(n.unique_name, **node_attrs)

        # Build up edges
        for e, attrs in typing.cast(typing.Mapping[typing.Tuple[StreamWrapper, StreamWrapper], dict], self._graph.edges()).items():  # noqa: E501

            edge_attrs = {}

            start_name = e[0].unique_name

            # Append the port if necessary
            if (has_ports(e[0], is_input=False)):
                start_name += f":d{attrs['start_port_idx']}"
            else:
                start_name += start_def_port

            end_name = e[1].unique_name

            if (has_ports(e[1], is_input=True)):
                end_name += f":u{attrs['end_port_idx']}"
            else:
                end_name += end_def_port

            # Now we only want to show the type label in some scenarios:
            # 1. If there is only one edge between two nodes, draw type in middle "label"
            # 2. For port with an edge, only draw that port's type once (using index 0 of the senders/receivers)
            start_port_idx = int(attrs['start_port_idx'])
            end_port_idx = int(attrs['end_port_idx'])

            out_port = e[0].output_ports[start_port_idx]
            in_port = e[1].input_ports[end_port_idx]

            # Check for situation #1
            if (len(in_port._input_senders) == 1 and len(out_port._output_receivers) == 1
                    and (in_port.in_type == out_port.out_type)):

                edge_attrs["label"] = pretty_print_type_name(in_port.in_type)
            else:
                rec_idx = out_port._output_receivers.index(in_port)
                sen_idx = in_port._input_senders.index(out_port)

                # Add type labels if available
                if (rec_idx == 0 and out_port.out_type is not None):
                    edge_attrs["taillabel"] = pretty_print_type_name(out_port.out_type)

                if (sen_idx == 0 and in_port.in_type is not None):
                    edge_attrs["headlabel"] = pretty_print_type_name(in_port.in_type)

            gv_graph.edge(start_name, end_name, **edge_attrs)

        file_format = os.path.splitext(filename)[-1].replace(".", "")

        viz_binary = gv_graph.pipe(format=file_format)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:
            f.write(viz_binary)

    async def _do_run(self):
        """
        This function sets up the current asyncio loop, builds the pipeline, and awaits on it to complete.
        """
        loop = asyncio.get_running_loop()

        def error_handler(_, context: dict):

            msg = "Unhandled exception in async loop! Exception: \n{}".format(context["message"])
            exception = context.get("exception", Exception())

            logger.critical(msg, exc_info=exception)

        loop.set_exception_handler(error_handler)

        exit_count = 0

        # Handles Ctrl+C for graceful shutdown
        def term_signal():

            nonlocal exit_count
            exit_count = exit_count + 1

            if (exit_count == 1):
                tqdm.write("Stopping pipeline. Please wait... Press Ctrl+C again to kill.")
                self.stop()
            else:
                tqdm.write("Killing")
                exit(1)

        for s in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(s, term_signal)

        try:
            self.build_and_start()

            # Wait for completion
            await self.join()

        except KeyboardInterrupt:
            tqdm.write("Stopping pipeline. Please wait...")

            # Stop the pipeline
            self.stop()

            # Wait again for nice completion
            await self.join()

        finally:
            # Shutdown the async generator sources and exit
            logger.info("====Pipeline Complete====")

    def run(self):
        """
        This function makes use of asyncio features to keep the pipeline running indefinitely.
        """

        # Use asyncio.run() to launch the pipeline. This creates and destroys an event loop so re-running a pipeline in
        # the same process wont fail
        asyncio.run(self._do_run())
