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
import logging
import os
import pickle
import time
import typing

import click
import mrc
import mrc.core.operators as ops
import pandas as pd
import pymilvus
import requests_cache
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.rss_source_stage import RSSSourceStage

logger = logging.getLogger(f"morpheus.{__name__}")


class RSSDownloadStage(SinglePortStage):

    def __init__(self, c: Config, *, chunk_size, link_column: str = "link"):
        super().__init__(c)

        self._link_column = link_column
        self._chunk_size = chunk_size

        self._cache_dir = "./.cache/llm/rss/"

        # Ensure the directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size,
                                                             chunk_overlap=self._chunk_size // 10,
                                                             length_function=len)

        self._session = requests_cache.CachedSession(os.path.join("./.cache/http", "RSSDownloadStage.sqlite"),
                                                     backend="sqlite")

        self._session.headers.update({
            "User-Agent":
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        })

    @property
    def name(self) -> str:
        """Returns the name of this stage."""
        return "rss-download"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        node = builder.make_node(self.unique_name, ops.map(self._download_and_split))
        node.launch_options.pe_count = self._config.num_threads

        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]

    def _download_and_split(self, msg: MessageMeta) -> MessageMeta:

        from bs4 import BeautifulSoup
        from langchain.schema import Document
        from newspaper import Article

        # Convert the dataframe into a list of dictionaries
        df_pd: pd.DataFrame = msg.df.to_pandas()
        df_dicts = df_pd.to_dict(orient="records")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

        final_rows: list[dict] = []

        for row in df_dicts:
            url = row[self._link_column]

            try:
                # Try to get the page content
                response = self._session.get(url)

                if (not response.ok):
                    logger.warning(
                        f"Error downloading document from URL '{url}'. Returned code: {response.status_code}. With reason: '{response.reason}'"
                    )
                    continue

                raw_html = response.text

                soup = BeautifulSoup(raw_html, "html.parser")

                text = soup.get_text(strip=True)

                # article = Article(url)
                # article.download()
                # article.parse()
                # print(article.text)
                # text = article.text

                split_text = splitter.split_text(text)

                for text in split_text:
                    r = row.copy()
                    r.update({"page_content": text})
                    final_rows.append(r)

                logger.debug(f"Processed page: '{url}'. Cache hit: {response.from_cache}")

            except ValueError as e:
                logger.error(f"Error parsing document: {e}")
                continue

        return MessageMeta(cudf.from_pandas(pd.DataFrame(final_rows)))


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
@click.option("--pre_calc_embeddings",
              is_flag=True,
              default=False,
              help="Whether to pre-calculate the embeddings using Triton")
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
             pre_calc_embeddings,
             isolate_embeddings,
             use_cache):

    from morpheus.config import Config
    from morpheus.config import CppConfig
    from morpheus.config import PipelineModes
    from morpheus.pipeline.linear_pipeline import LinearPipeline
    from morpheus.stages.general.monitor_stage import MonitorStage
    from morpheus.stages.general.trigger_stage import TriggerStage
    from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
    from morpheus.stages.output.write_to_vector_db import WriteToVectorDBStage
    from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
    from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

    from ..common.arxiv_source import ArxivSource

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

    rss_urls = [
        "https://www.theregister.com/security/headlines.atom",
        "https://isc.sans.edu/dailypodcast.xml",
        "https://threatpost.com/feed/",
        "http://feeds.feedburner.com/TheHackersNews?format=xml",
        "https://www.bleepingcomputer.com/feed/",
        "https://therecord.media/feed/",
        "https://blog.badsectorlabs.com/feeds/all.atom.xml",
        "https://krebsonsecurity.com/feed/",
        "https://www.darkreading.com/rss_simple.asp",  # "https://blog.talosintelligence.com/feeds/posts/default",
        "https://blog.malwarebytes.com/feed/",
        "https://msrc.microsoft.com/blog/feed",
        # "https://www.f-secure.com/en/business/resources/newsroom/blog",
        # "https://otx.alienvault.com/api/v1/pulses.rss",
        "https://securelist.com/feed",
        "https://www.crowdstrike.com/blog/feed/",
        "https://threatconnect.com/blog/rss/",
        "https://news.sophos.com/en-us/feed/",  # "https://www.trendmicro.com/vinfo/us/security/news/feed",
        "https://www.us-cert.gov/ncas/current-activity.xml",
        "https://www.csoonline.com/feed",
        "https://www.cyberscoop.com/feed",
        "https://research.checkpoint.com/feed",
        "https://feeds.fortinet.com/fortinet/blog/threat-research",
        # "https://www.proofpoint.com/us/blog/threat-insight",
        "https://www.mcafee.com/blogs/rss",
        # "https://symantec-enterprise-blogs.security.com/blogs/threat-intelligence",
        "https://www.digitalshadows.com/blog-and-research/rss.xml",
        "https://www.nist.gov/news-events/cybersecurity/rss.xml",
        "https://www.sentinelone.com/blog/rss/",
        "https://www.bitdefender.com/blog/api/rss/labs/",
        "https://www.welivesecurity.com/feed/",
        "https://unit42.paloaltonetworks.com/feed/",
        "https://mandiant.com/resources/blog/rss.xml",
        "https://www.wired.com/feed/category/security/latest/rss",
        "https://www.wired.com/feed/tag/ai/latest/rss",
        # "https://blog.talosintelligence.com/",
        # "https://blog.fox-it.com/",
        # "https://blog.qualys.com/vulnerabilities-threat-research",
        "https://blog.google/threat-analysis-group/rss/",  # "https://securityintelligence.com/x-force/",
        "https://intezer.com/feed/",  # "https://securelist.com",
        # "https://socprime.com/blog/",
    ]

    # add doca source stage
    # pipeline.set_source(FileSourceStage(config, filename=input_file, repeat=1))
    # pipe.set_source(ArxivSource(config))
    pipe.set_source(RSSSourceStage(config, feed_input=rss_urls, batch_size=128))

    pipe.add_stage(MonitorStage(config, description="Source rate", unit='pages'))

    pipe.add_stage(RSSDownloadStage(config, chunk_size=model_fea_length))

    pipe.add_stage(MonitorStage(config, description="Download rate", unit='pages'))

    if (isolate_embeddings):
        pipe.add_stage(TriggerStage(config))

    if (pre_calc_embeddings):

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

    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name="Arxiv",
                             resource_kwargs=milvus_resource_kwargs,
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

        loader = ConfluenceLoader(
            url="https://confluence.nvidia.com",
            token=os.environ.get("CONFLUENCE_API_KEY", None),
        )

        documents = loader.load(space_key="PRODSEC", max_pages=2000)

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
