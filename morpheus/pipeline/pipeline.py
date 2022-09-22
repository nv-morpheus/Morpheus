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

from collections import defaultdict
from functools import partial
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

        # Complete set of nodes across segments in this pipeline
        self._stages: typing.Set[Stage] = set()
        # Complete set of sources across segments in this pipeline
        self._sources: typing.Set[SourceStage] = set()
        # Dictionary containing segment information for this pipeline
        self._segments: typing.Dict = defaultdict(lambda: {
            "nodes": set(),
            "ingress_ports": [],
            "egress_ports": []
        })

        self._exec_options = srf.Options()
        self._exec_options.topology.user_cpuset = "0-{}".format(c.num_threads - 1)
        self._exec_options.engine_factories.default_engine_type = srf.core.options.EngineType.Thread

        # Set the default channel size
        srf.Config.default_channel_size = c.edge_buffer_size

        self.batch_size = c.pipeline_batch_size

        # self._graph = networkx.DiGraph()
        self._segment_graphs = defaultdict(lambda: networkx.DiGraph())

        self._is_built = False
        self._is_build_complete = False
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

    def add_node(self, node: StreamWrapper, segment_id: str = "main"):

        assert node._pipeline is None or node._pipeline is self, "A stage can only be added to one pipeline at a time"

        segment_nodes = self._segments[segment_id]["nodes"]
        segment_graph = self._segment_graphs[segment_id]

        # Add to list of stages if it's a stage, not a source
        if (isinstance(node, Stage)):
            segment_nodes.add(node)
            self._stages.add(node)
        elif (isinstance(node, SourceStage)):
            segment_nodes.add(node)
            self._sources.add(node)
        else:
            raise NotImplementedError("add_node() failed. Unknown node type: {}".format(type(node)))

        node._pipeline = self

        # self._graph.add_node(node)
        segment_graph.add_node(node)

    def add_edge(self, start: typing.Union[StreamWrapper, Sender], end: typing.Union[Stage, Receiver],
                 segment_id: str = "main"):

        if (isinstance(start, StreamWrapper)):
            start_port = start.output_ports[0]
        elif (isinstance(start, Sender)):
            start_port = start

        if (isinstance(end, Stage)):
            end_port = end.input_ports[0]
        elif (isinstance(end, Receiver)):
            end_port = end

        # Ensure both are added to this pipeline
        self.add_node(start_port.parent, segment_id)
        self.add_node(end_port.parent, segment_id)

        start_port._output_receivers.append(end_port)
        end_port._input_senders.append(start_port)

        segment_graph = self._segment_graphs[segment_id]
        segment_graph.add_edge(start_port.parent,
                               end_port.parent,
                               start_port_idx=start_port.port_number,
                               end_port_idx=end_port.port_number)

    def add_ingress(self, segment_id: str, ingress_tuple: typing.Tuple):
        segment_data = self._segments[segment_id]["ingress_ports"]
        segment_data.append(ingress_tuple)
        print(f"Appending ingress tuple: {ingress_tuple}")

    def add_egress(self, segment_id: str, egress_tuple: typing.Tuple):
        segment_data = self._segments[segment_id]["egress_ports"]
        segment_data.append(egress_tuple)
        print(f"Appending egress tuple: {egress_tuple}")

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

        def inner_build(builder: srf.Builder, segment_id: str):
            segment_graph = self._segment_graphs[segment_id]

            # Get the list of stages and source
            # source_and_stages: typing.List[StreamWrapper] = list(self._sources) + list(self._stages)

            # This should be a BFS search from all source nodes
            for stage in networkx.topological_sort(self._segment_graphs[segment_id]):
                stage.build(builder, do_propagate=False)

            # Now loop over stages
            # for s in source_and_stages:
            #
            #    if (s.can_build()):
            #        s.build(builder)

            # if (not all([x.is_built for x in source_and_stages])):
            #    # raise NotImplementedError("Circular pipelines are not yet supported!")
            #    logger.warning("Circular pipeline detected! Building with reduced constraints")

            #    for s in source_and_stages:

            #        if (s.can_build(check_ports=True)):
            #            s.build()

            # if (not all([x.is_built for x in source_and_stages])):
            if (not all([x.is_built for x in segment_graph.nodes()])):
                raise RuntimeError("Could not build pipeline. Ensure all types can be determined")

            # Finally, execute the link phase (only necessary for circular pipelines)
            # for s in source_and_stages:
            for s in segment_graph.nodes():
                for p in s.input_ports:
                    p.link()

            # logger.info("====Building Pipeline Complete!====")
            # self._is_build_complete = True

            ## Finally call _on_start
            # self._on_start()

        logger.info("====Building Pipeline====")
        for segment_id in self._segments.keys():
            # segment_stages is a set of stages
            segment_ingress_ports = self._segments[segment_id]["ingress_ports"]
            segment_egress_ports = self._segments[segment_id]["egress_ports"]
            segment_inner_build = partial(inner_build, segment_id=segment_id)


            print(segment_id)
            print(segment_ingress_ports)
            print(segment_egress_ports)
            print(segment_inner_build)
            self._srf_pipeline.make_segment(segment_id, segment_ingress_ports,
                                            segment_egress_ports, segment_inner_build)

        logger.info("====Building Pipeline Complete!====")
        self._is_build_complete = True

        # Finally call _on_start
        self._on_start()

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

    async def build_and_start(self):

        if (not self.is_built):
            try:
                self.build()
            except Exception:
                logger.exception("Error occurred during Pipeline.build(). Exiting.", exc_info=True)
                return

        await self.async_start()

        self.start()

    async def async_start(self):

        # Loop over all stages and call on_start if it exists
        for s in self._stages:
            await s.start_async()

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

    # TODO(Devin) : Probably going to break
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

        if not self._is_build_complete:
            raise RuntimeError("Pipeline.visualize() requires that the Pipeline has been started before generating "
                               "the visualization. Please call Pipeline.start(), Pipeline.build_and_start() or "
                               "Pipeline.run() before calling Pipeline.visualize(). This is a known issue and will "
                               "be fixed in a future release.")

        # Now build up the nodes
        # TODO(Devin)
        segment_id = "segment_0"
        for n, attrs in typing.cast(typing.Mapping[StreamWrapper, dict],
                                    self._segment_graphs[segment_id].nodes).items():
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
        # TODO(Devin)
        segment_id = "segment_0"
        for e, attrs in typing.cast(typing.Mapping[typing.Tuple[StreamWrapper, StreamWrapper], dict],
                                    self._segment_graphs[segment_id].edges()).items():  # noqa: E501

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
            await self.build_and_start()

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
