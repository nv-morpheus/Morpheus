# Copyright (c) 2022, NVIDIA CORPORATION, All rights reserved.
"""Module to provide pipeline stage the reports results to an Elasticsearch DB."""
import logging
import typing
from ssl import create_default_context

import mrc
import mrc.core.operators as ops
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, parallel_bulk
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from pyarrow.lib import ArrowException

logger = logging.getLogger(f"morpheus.{__name__}")

@register_stage("to-elasticsearch")
class WriteToElasticsearchStage(SinglePortStage):
    """Pipeline stage that writes results to Elasticsearch DB."""

    def __init__(self,
                 c: Config,
                 elastic_index: str,
                 elastic_hosts: list,
                 elastic_ports: list,
                 elastic_user: str,
                 elastic_password: str,
                 elastic_cacrt: str = "",
                 elastic_scheme: str = "http",
                 num_threads: int = 16):
        # pylint: disable=too-many-arguments

        super().__init__(c)
        self._index = elastic_index
        self._num_threads = num_threads

        if elastic_scheme == "https":
            context = create_default_context(cafile=elastic_cacrt)
            self._es = Elasticsearch(
                self._build_hosts(elastic_hosts, elastic_ports, elastic_scheme),
                http_auth=(elastic_user, elastic_password),
                ssl_context=context,
            )
        else:
            self._es = Elasticsearch(
                self._build_hosts(elastic_hosts, elastic_ports, elastic_scheme),
                http_auth=(elastic_user, elastic_password),
            )

    def _build_hosts(self, elastic_hosts, elastic_ports, elastic_scheme):

        return [
            {
                'host': host,
                'port': int(port),
                'scheme': elastic_scheme
            } for host, port in zip(elastic_hosts.split(','), elastic_ports.split(','))
        ]

    def supports_cpp_node(self):
        return False

    @property
    def name(self) -> str:
        return "to-elasticsearch"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.
        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.
        """
        return (MessageMeta, )

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            def on_data(message: MessageMeta):
            
                try:
                    records = serializers.df_to_json(message.df, strip_newlines=True)
                except ArrowException as e:
                    return message
                
                actions = [{"_index": self._index, "_source": record} for record in records]                      
                
                for okay, info in streaming_bulk(self._es,
                                                 actions=actions,
                                                 raise_on_exception=False,):

                    if not okay:
                        logger.error("Error writing to ElasticSearch: %s", str(info))
                        sub.on_error(info)
                        
                return message

            obs.pipe(ops.map(on_data)).subscribe(sub)

        node = builder.make_node_full(self.unique_name, node_fn)

        # cpu cores
        node.launch_options.pe_count = 1
        # threads per core
        node.launch_options.engines_per_pe = self._num_threads
        
        builder.make_edge(input_stream[0], node)

        return node, MessageMeta
