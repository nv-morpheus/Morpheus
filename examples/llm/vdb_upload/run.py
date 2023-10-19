# Copyright (c) 2023, NVIDIA CORPORATION.
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
import functools
import logging
import os
import pickle
import time
import typing

import click

logger = logging.getLogger(f"morpheus.{__name__}")


def _build_milvus_config(embedding_size: int):
    import pymilvus

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
                pymilvus.FieldSchema(name="title",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The title of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="link",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The URL of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="summary",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The summary of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="page_content",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="A chunk of text from the RSS Page",
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


def _build_rss_urls():
    return [
        "https://www.theregister.com/security/headlines.atom",
        "https://isc.sans.edu/dailypodcast.xml",
        "https://threatpost.com/feed/",
        "http://feeds.feedburner.com/TheHackersNews?format=xml",
        "https://www.bleepingcomputer.com/feed/",
        "https://therecord.media/feed/",
        "https://blog.badsectorlabs.com/feeds/all.atom.xml",
        "https://krebsonsecurity.com/feed/",
        "https://www.darkreading.com/rss_simple.asp",
        "https://blog.malwarebytes.com/feed/",
        "https://msrc.microsoft.com/blog/feed",
        "https://securelist.com/feed",
        "https://www.crowdstrike.com/blog/feed/",
        "https://threatconnect.com/blog/rss/",
        "https://news.sophos.com/en-us/feed/",
        "https://www.us-cert.gov/ncas/current-activity.xml",
        "https://www.csoonline.com/feed",
        "https://www.cyberscoop.com/feed",
        "https://research.checkpoint.com/feed",
        "https://feeds.fortinet.com/fortinet/blog/threat-research",
        "https://www.mcafee.com/blogs/rss",
        "https://www.digitalshadows.com/blog-and-research/rss.xml",
        "https://www.nist.gov/news-events/cybersecurity/rss.xml",
        "https://www.sentinelone.com/blog/rss/",
        "https://www.bitdefender.com/blog/api/rss/labs/",
        "https://www.welivesecurity.com/feed/",
        "https://unit42.paloaltonetworks.com/feed/",
        "https://mandiant.com/resources/blog/rss.xml",
        "https://www.wired.com/feed/category/security/latest/rss",
        "https://www.wired.com/feed/tag/ai/latest/rss",
        "https://blog.google/threat-analysis-group/rss/",
        "https://intezer.com/feed/",
    ]


@click.group(name=__name__)
def run():
    pass


@run.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=64,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_fea_length",
    default=256,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--embedding_size",
    default=384,
    type=click.IntRange(min=1),
    help="Output size of the embedding model",
)
@click.option(
    "--input_file",
    default="output.csv",
    help="The path to input event stream",
)
@click.option(
    "--output_file",
    default="output.csv",
    help="The path to the file where the inference output will be saved.",
)
@click.option("--server_url", required=True, default='192.168.0.69:8000', help="Tritonserver url")
@click.option(
    "--model_name",
    required=True,
    default='all-mpnet-base-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option("--isolate_embeddings",
              is_flag=True,
              default=False,
              help="Whether to pre-calculate the embeddings using Triton")
@click.option("--use_cache",
              type=click.Path(file_okay=True, dir_okay=False),
              default=None,
              help="What cache to use for the confluence documents")
def pipeline(num_threads,
             pipeline_batch_size,
             model_max_batch_size,
             model_fea_length,
             embedding_size,
             input_file,
             output_file,
             server_url,
             model_name,
             isolate_embeddings,
             use_cache):

    from morpheus.config import Config
    from morpheus.config import CppConfig
    from morpheus.config import PipelineModes
    from morpheus.pipeline.linear_pipeline import LinearPipeline
    from morpheus.stages.general.monitor_stage import MonitorStage
    from morpheus.stages.general.trigger_stage import TriggerStage
    from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
    from morpheus.stages.input.rss_source_stage import RSSSourceStage
    from morpheus.stages.output.write_to_vector_db import WriteToVectorDBStage
    from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
    from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

    from ..common.web_scraper_stage import WebScraperStage

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    config.class_labels = [str(i) for i in range(embedding_size)]

    pipe = LinearPipeline(config)

    # add doca source stage
    pipe.set_source(RSSSourceStage(config, feed_input=_build_rss_urls(), batch_size=128))

    pipe.add_stage(MonitorStage(config, description="Source rate", unit='pages'))

    pipe.add_stage(WebScraperStage(config, chunk_size=model_fea_length))

    pipe.add_stage(MonitorStage(config, description="Download rate", unit='pages'))

    if (isolate_embeddings):
        pipe.add_stage(TriggerStage(config))

    # add deserialize stage
    pipe.add_stage(DeserializeStage(config))

    # add preprocessing stage
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file="data/bert-base-uncased-hash.txt",
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='page_content'))

    pipe.add_stage(MonitorStage(config, description="Tokenize rate", unit='events', delayed_start=True))

    pipe.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url="localhost:8001",
                             force_convert_inputs=True,
                             use_shared_memory=True))
    pipe.add_stage(MonitorStage(config, description="Inference rate", unit="events", delayed_start=True))

    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name="RSS",
                             resource_kwargs=_build_milvus_config(embedding_size=embedding_size),
                             recreate=True,
                             service="milvus",
                             uri="http://localhost:19530"))

    pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    start_time = time.time()

    pipe.run()

    duration = time.time() - start_time

    print(f"Total duration: {duration:.2f} seconds")


@run.command()
@click.option(
    "--model_name",
    required=True,
    default='all-mpnet-base-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option(
    "--save_cache",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Location to save the cache to",
)
def chain(model_name, save_cache):
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores.milvus import Milvus

    from morpheus.utils.logging_timer import log_time

    with log_time(msg="Seeding with chain took {duration} ms. {rate_per_sec} docs/sec", log_fn=logger.debug) as l:

        from langchain.document_loaders.rss import RSSFeedLoader

        loader = RSSFeedLoader(urls=_build_rss_urls())

        documents = loader.load()

        if (save_cache is not None):
            with open(save_cache, "wb") as f:
                pickle.dump(documents, f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len)

        documents = text_splitter.split_documents(documents)

        l.count = len(documents)

        logger.info(f"Loaded %s documents", len(documents))

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={
                # 'normalize_embeddings': True, # set True to compute cosine similarity
                "batch_size": 100,
            })

        with log_time(msg="Adding to Milvus took {duration} ms. Doc count: {count}. {rate_per_sec} docs/sec",
                      count=l.count,
                      log_fn=logger.debug):

            Milvus.from_documents(documents, embeddings, collection_name="LangChain", drop_old=True)


def _save_model(model, sample_input: dict, output_model_path: str):
    import torch

    try:
        import onnx
    except ImportError:
        raise RuntimeError("Please install onnx to use this feature. Run `mamba install -c conda-forge onnx`")

    device = torch.device("cuda")

    # input_ids = torch.ones(batch_size, max_seq_length, dtype=torch.int32).to(device)
    # input_mask = torch.ones(batch_size, max_seq_length, dtype=torch.int32).to(device)

    # Ensure our input is a dictionary, not a batch encoding
    args = ({k: v for k, v in sample_input.items()}, )

    import inspect
    inspect.signature(model.forward)

    torch.onnx.export(
        model,
        args,
        output_model_path,
        opset_version=13,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {
                0: 'batch_size',
                1: "seq_length",
            },  # variable length axes
            'attention_mask': {
                0: 'batch_size',
                1: "seq_length",
            },
            'output': {
                0: 'batch_size',
            }
        },
        verbose=False)

    onnx_model = onnx.load(output_model_path)

    onnx.checker.check_model(onnx_model)


@run.command()
@click.option(
    "--model_name",
    required=True,
    default='all-mpnet-base-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option(
    "--model_seq_length",
    default=512,
    type=click.IntRange(min=1),
    help="Accepted input size of the text tokens",
)
@click.option(
    "--max_batch_size",
    default=256,
    type=click.IntRange(min=1),
    help="Max batch size for the model config",
)
@click.option(
    "--triton_repo",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory of the Triton Model Repo where the model will be saved",
)
@click.option(
    "--output_model_name",
    default=None,
    help="Overrides the model name that is used in triton. Defaults to `model_name`",
)
def export_triton_model(model_name, model_seq_length, max_batch_size, triton_repo, output_model_name):

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    from transformers import AutoTokenizer
    from tritonclient.grpc.model_config_pb2 import DataType
    from tritonclient.grpc.model_config_pb2 import ModelConfig
    from tritonclient.grpc.model_config_pb2 import ModelInput
    from tritonclient.grpc.model_config_pb2 import ModelOptimizationPolicy
    from tritonclient.grpc.model_config_pb2 import ModelOutput
    from tritonclientutils import np_to_triton_dtype

    if (output_model_name is None):
        output_model_name = model_name

    class CustomTokenizer(torch.nn.Module):

        def __init__(self, model_name: str):
            super().__init__()

            import inspect

            from sentence_transformers import SentenceTransformer
            from transformers.models.bert.modeling_bert import BertModel

            self.inner_model = AutoModel.from_pretrained(model_name)
            # self.inner_model = SentenceTransformer(model_name)

            if (isinstance(self.inner_model, SentenceTransformer)):
                self._output_dim = self.inner_model.get_sentence_embedding_dimension()
            elif (isinstance(self.inner_model, BertModel)):
                self._output_dim = self.inner_model.config.hidden_size

            sig = inspect.signature(self.inner_model.forward)

            ordered_list_keys = list(sig.parameters.keys())
            if ordered_list_keys[0] == "self":
                ordered_list_keys = ordered_list_keys[1:]

            # Save the idx of the attention mask because exporting prefers arguments over kwargs
            self._attention_mask_idx = ordered_list_keys.index("attention_mask")

            # Wrap the original function so the export can find the original signature
            @functools.wraps(self.inner_model.forward)
            def forward(*args, **kwargs):
                return self._forward(*args, **kwargs)

            self.forward = forward

        @property
        def output_dim(self):
            return self._output_dim

        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(self, model_output, attention_mask):
            # Adapted from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
            # First element of model_output contains all token embeddings
            last_hidden_state = model_output["last_hidden_state"]  # [batch_size, seq_length, hidden_size]

            alternate = True

            if (alternate):
                last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            # Transpose to make broadcasting possible
            last_hidden_state = torch.transpose(last_hidden_state, 0, 2)  # [hidden_size, seq_length, batch_size]

            input_mask_expanded = torch.transpose(attention_mask.unsqueeze(-1).float(), 0,
                                                  2)  # [1, seq_length, batch_size]

            num = torch.sum(last_hidden_state * input_mask_expanded, 1)  # [hidden_size, batch_size]
            denom = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # [1, batch_size]

            return torch.transpose(num / denom, 0, 1)  # [batch_size, hidden_size]

        def normalize(self, embeddings):

            alternate = False

            if (alternate):
                return F.normalize(embeddings, p=2, dim=1)

            # Use the same trick here to broadcast to avoid using the expand operator which breaks dynamic axes
            denom = torch.transpose(embeddings.norm(2, 1, keepdim=True).clamp_min(1e-12), 0, 1)

            return torch.transpose(torch.transpose(embeddings, 0, 1) / denom, 0, 1)

        def _forward(self, *args, **kwargs):

            if ("attention_mask" in kwargs):
                attention_mask = kwargs["attention_mask"]
            elif (len(args) > self._attention_mask_idx):
                # Lookup from positional
                attention_mask = args[self._attention_mask_idx]
            else:
                raise RuntimeError("Cannot determine attention mask")

            model_outputs = self.inner_model(*args, **kwargs)

            sentence_embeddings = self.mean_pooling(model_outputs, attention_mask)

            sentence_embeddings = self.normalize(sentence_embeddings)

            return sentence_embeddings

    device = torch.device("cuda")

    model_name = f'{model_name}'

    model = CustomTokenizer(model_name)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_texts = [
        "This is text one which is longer",
        "This is text two",
    ]

    model_input_names = ["input_ids", "attention_mask"]

    sample_input = tokenizer(test_texts,
                             max_length=model_seq_length,
                             padding="max_length",
                             truncation=True,
                             return_token_type_ids=False,
                             return_tensors="pt").to(device)

    test_output = model(**(sample_input.to(device))).detach()

    output_model_dir = os.path.join(triton_repo, output_model_name)

    # Make sure we create the directory if it does not exist
    os.makedirs(output_model_dir, exist_ok=True)

    # Make the config file
    config = ModelConfig()

    config.name = output_model_name
    config.platform = "onnxruntime_onnx"
    config.max_batch_size = max_batch_size

    for input_name, input_data in sample_input.data.items():

        config.input.append(
            ModelInput(
                name=input_name,
                data_type=DataType.Value(f"TYPE_{np_to_triton_dtype(input_data.cpu().numpy().dtype)}"),
                dims=[input_data.shape[1]],
            ))

    config.output.append(
        ModelOutput(
            name="output",
            data_type=DataType.Value(f"TYPE_{np_to_triton_dtype(test_output.cpu().numpy().dtype)}"),
            dims=[test_output.shape[1]],
        ))

    def _powers_of_2(max_val: int):
        val = 1

        while (val <= max_val):
            yield val
            val *= 2

    config.dynamic_batching.preferred_batch_size.extend([x for x in _powers_of_2(max_batch_size)])
    config.dynamic_batching.max_queue_delay_microseconds = 50000

    config.optimization.execution_accelerators.gpu_execution_accelerator.extend([
        ModelOptimizationPolicy.ExecutionAccelerators.Accelerator(name="tensorrt",
                                                                  parameters={
                                                                      "precision_mode": "FP16",
                                                                      "max_workspace_size_bytes": "1073741824",
                                                                  })
    ])

    config_path = os.path.join(output_model_dir, "config.pbtxt")

    with open(config_path, "w") as f:
        f.write(str(config))

    model_version_dir = os.path.join(output_model_dir, "1")

    os.makedirs(model_version_dir, exist_ok=True)

    output_model_path = os.path.join(model_version_dir, "model.onnx")

    _save_model(model, sample_input, output_model_path=output_model_path)

    logger.info("Created Triton Model at %s", output_model_dir)
