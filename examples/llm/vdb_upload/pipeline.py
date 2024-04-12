# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import typing

from vdb_upload.helper import process_vdb_sources

from morpheus.config import Config
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

logger = logging.getLogger(__name__)


def pipeline(pipeline_config: Config,
             source_config: typing.List,
             vdb_config: typing.Dict,
             embeddings_config: typing.Dict,
             tokenizer_config: typing.Dict) -> float:
    """
    Sets up and runs a data processing pipeline based on provided configurations.

    Parameters
    ----------
    pipeline_config : Dict
        General configuration for the pipeline, including number of threads and batch sizes.
    source_config : List[Dict]
        Configuration for data sources, specifying the type of sources to use (e.g., 'rss', 'filesystem') and their
        individual settings.
    vdb_config : Dict
        Configuration settings for the vector database, detailing how vectors should be stored, queried, and managed.
    embeddings_config : Dict
        Configuration for generating embeddings, including model name, embedding size, and any model-specific settings.
    tokenizer_config : Dict
        Configuration for the tokenizer, specifying how text should be tokenized before embedding. Includes tokenizer
        model and settings.

    Returns
    -------
    float
        The start time of the pipeline execution, typically represented as a timestamp.
    """

    isolate_embeddings = embeddings_config.get('isolate_embeddings', False)

    pipe = Pipeline(pipeline_config)

    # vdb_sources = process_vdb_sources(pipe, pipeline_config, source_config)

    from morpheus.stages.doca.doca_convert_stage import DocaConvertStage
    from morpheus.stages.doca.doca_source_stage import DocaSourceStage
    from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

    source_stage = pipe.add_stage(DocaSourceStage(pipeline_config, "ca:00.0", "17:00.0", "udp"))
    convert_stage = pipe.add_stage(DocaConvertStage(pipeline_config, False))
    deserialize_stage = pipe.add_stage(DeserializeStage(pipeline_config))

    # test_mon = pipe.add_stage(MonitorStage(pipeline_config, description="DOCA GPUNetIO Source rate", unit='pkts'))

    # pipe.add_edge(source_stage, test_mon)
    # pipe.add_edge(test_mon, convert_stage)
    # pipe.add_edge(convert_stage, deserialize_stage)

    pipe.add_edge(source_stage, convert_stage)
    pipe.add_edge(convert_stage, deserialize_stage)

    vdb_sources = [deserialize_stage]

    trigger = None
    if (isolate_embeddings):
        trigger = pipe.add_stage(TriggerStage(pipeline_config))

    nlp_stage = pipe.add_stage(PreprocessNLPStage(pipeline_config, **tokenizer_config.get("model_kwargs", {})))

    monitor_1 = pipe.add_stage(
        MonitorStage(pipeline_config, description="Tokenize rate", unit='events', delayed_start=True))

    embedding_stage = pipe.add_stage(TritonInferenceStage(pipeline_config, **embeddings_config.get('model_kwargs', {})))

    monitor_2 = pipe.add_stage(
        MonitorStage(pipeline_config, description="Inference rate", unit="events", delayed_start=True))

    # vector_db = pipe.add_stage(WriteToVectorDBStage(pipeline_config, **vdb_config))

    # monitor_3 = pipe.add_stage(
    #     MonitorStage(pipeline_config, description="Upload rate", unit="events", delayed_start=True))

    # Connect the pipeline
    for source_output in vdb_sources:
        mon_stage = pipe.add_stage(
            MonitorStage(pipeline_config, description="Source Rate", unit="events", delayed_start=False))

        pipe.add_edge(source_output, mon_stage)

        if (isolate_embeddings):
            pipe.add_edge(mon_stage, trigger)
        else:
            pipe.add_edge(mon_stage, nlp_stage)

    if (isolate_embeddings):
        pipe.add_edge(trigger, nlp_stage)

    pipe.add_edge(nlp_stage, monitor_1)
    pipe.add_edge(monitor_1, embedding_stage)
    pipe.add_edge(embedding_stage, monitor_2)
    # pipe.add_edge(monitor_2, vector_db)
    # pipe.add_edge(vector_db, monitor_3)

    start_time = time.time()

    pipe.run()

    return start_time