# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

import click
import pymilvus

from morpheus._lib.messages import RawPacketMessage
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBService
from morpheus.service.vdb.utils import VectorDBServiceFactory
from morpheus.stages.doca.doca_convert_stage import DocaConvertStage
from morpheus.stages.doca.doca_source_stage import DocaSourceStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging


def build_milvus_service(embedding_size):

    milvus_resource_kwargs = {
        "index_conf": {
            "field_name": "embedding",
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 8,
                "efConstruction": 64,
            },
        },
        "schema_conf": {
            "enable_dynamic_field": True,
            "schema_fields": [
                pymilvus.FieldSchema(name="id",
                                     dtype=pymilvus.DataType.INT64,
                                     description="Primary key for the collection",
                                     is_primary=True,
                                     auto_id=True).to_dict(),
                pymilvus.FieldSchema(name="src_ip",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The packet source IP address",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="data",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The packet data",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="embedding",
                                     dtype=pymilvus.DataType.FLOAT_VECTOR,
                                     description="Embedding vectors",
                                     dim=embedding_size).to_dict(),
            ],
            "description": "Test collection schema"
        }
    }

    return milvus_resource_kwargs


@click.command()
@click.option(
    "--out_file",
    default="doca_output.csv",
    help="File in which to store output",
)
@click.option(
    "--nic_addr",
    help="NIC PCI Address",
    required=True,
)
@click.option(
    "--gpu_addr",
    help="GPU PCI Address",
    required=True,
)
def run_pipeline(out_file, nic_addr, gpu_addr):
    # Enable the default logger
    configure_logging(log_level=logging.DEBUG)

    CppConfig.set_should_use_cpp(True)

    config = Config()
    config.mode = PipelineModes.NLP
    config.pipeline_batch_size = 2048
    config.feature_length = 512

    # Below properties are specified by the command line
    # config.num_threads = 5

    pipeline = LinearPipeline(config)

    # add doca source stage
    pipeline.set_source(DocaSourceStage(config, nic_addr, gpu_addr, 'udp'))
    pipeline.add_stage(DocaConvertStage(config, False))

    pipeline.add_stage(MonitorStage(config, description="DOCA GPUNetIO Source rate", unit='pkts'))

    pipeline.add_stage(DeserializeStage(config))

    pipeline.add_stage(PreprocessNLPStage(config))

    pipeline.add_stage(
        TritonInferenceStage(config,
                             force_convert_inputs=True,
                             model_name="all-MiniLM-L6-v2",
                             server_url="localhost:8001",
                             use_shared_memory=True))
    pipeline.add_stage(MonitorStage(config, description="Embedding rate", unit='pkts'))

    pipeline.add_stage(
        WriteToVectorDBStage(config,
                             resource_name="vdb_doca",
                             batch_size=16896,
                             recreate=True,
                             service="milvus",
                             uri="http://localhost:19530",
                             resource_schemas={"vdb_doca": build_milvus_service(384)}))
    pipeline.add_stage(MonitorStage(config, description="Upload rate", unit='docs'))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()