# Copyright (c) 2021, NVIDIA CORPORATION.
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
"""
Contains configuration objects used to run pipeline and utilities.
"""

import dataclasses
import json
import logging
import typing
from enum import Enum

logger = logging.getLogger(__name__)


def auto_determine_bootstrap():
    """Auto determine bootstrap servers for kafka cluster."""
    import docker

    kafka_compose_name = "kafka-docker"

    docker_client = docker.from_env()
    bridge_net = docker_client.networks.get("bridge")
    bridge_ip = bridge_net.attrs["IPAM"]["Config"][0]["Gateway"]

    kafka_net = docker_client.networks.get(kafka_compose_name + "_default")

    bootstrap_servers = ",".join([
        c.ports["9092/tcp"][0]["HostIp"] + ":" + c.ports["9092/tcp"][0]["HostPort"] for c in kafka_net.containers
        if "9092/tcp" in c.ports
    ])

    # Use this version to specify the bridge IP instead
    bootstrap_servers = ",".join(
        [bridge_ip + ":" + c.ports["9092/tcp"][0]["HostPort"] for c in kafka_net.containers if "9092/tcp" in c.ports])

    logger.info("Auto determined Bootstrap Servers: {}".format(bootstrap_servers))

    return bootstrap_servers


@dataclasses.dataclass
class ConfigBase():
    """This is the base class for pipeline configuration."""
    pass


@dataclasses.dataclass
class ConfigOnnxToTRT(ConfigBase):
    """
    Configuration class for the OnnxToTRT migration process.

    Parameters
    ----------
    input_model : str
        Input model file.
    output_model : str
        Output model file.
    batches : typing.List[typing.Tuple[int, int]]
        Batches of input data
    seq_length : int
        Sequence length
    max_workspace_size : int
        Maximum workspace size, default 16000 MB

    """
    input_model: str = None
    output_model: str = None
    batches: typing.List[typing.Tuple[int, int]] = dataclasses.field(default_factory=list)
    seq_length: int = None
    max_workspace_size: int = 16000  # In MB


@dataclasses.dataclass
class ConfigDask(ConfigBase):
    """
    Pipeline Dask configuration class.

    Parameters
    ----------
    use_processes : bool
        Not currently used.

    """
    use_processes: bool = False


class PipelineModes(str, Enum):
    """The type of usecases that can be executed by the pipeline is determined by the enum."""
    OTHER = "OTHER"
    NLP = "NLP"
    FIL = "FIL"


@dataclasses.dataclass
class Config(ConfigBase):
    """
    Pipeline configuration class.

    Parameters
    ----------
    debug : bool
        Flag to run pipeline in debug mode. Default value is False
    log_level : int
        Specifies the log level and above to output. Must be one of the available levels in the `logging` module.
    mode : PipelineModes
        Determines type of pipeline Ex: FIL or NLP. Use `OTHER` for custom pipelines.
    feature_length : int
        Specifies the dimension of the second axis for messages in the pipeline. For NLP this is the sequence length.
        For FIL this is the number of input features
    pipeline_batch_size : int
        Determines number of messages per batch. Default value is 256
    num_threads : int
        Number threads to process each batch of messages. Default value is 1
    model_max_batch_size : 8
        In a single batch, the maximum number of messages to send to the model for inference. Default value is
        8
    use_dask : bool
        Determines if the pipeline should be executed using the Dask scheduler. Default value is False

    """
    # Flag to indicate we are creating a static instance. Prevents raising an error on creation
    __is_creating: typing.ClassVar[bool] = False

    # Default should never be changed and is used to initialize the CLI
    __default: typing.ClassVar["Config"] = None
    # Singleton instance of the Config
    __instance: typing.ClassVar["Config"] = None

    # Whether in Debug mode.
    debug: bool = False
    log_level: int = logging.WARN
    log_config_file: str = None

    mode: PipelineModes = PipelineModes.OTHER

    feature_length: int = 256
    pipeline_batch_size: int = 256
    num_threads: int = 1
    model_max_batch_size: int = 8

    use_dask: bool = False

    dask: ConfigDask = dataclasses.field(default_factory=ConfigDask)

    @staticmethod
    def default() -> "Config":
        if Config.__default is None:
            try:
                Config.__is_creating = True
                Config.__default = Config()
            finally:
                Config.__is_creating = False

        return Config.__default

    @classmethod
    def get(cls) -> "Config":
        if cls.__instance is None:
            try:
                cls.__is_creating = True
                cls.__instance = Config()
            finally:
                cls.__is_creating = False

        return cls.__instance

    def __post_init__(self):
        # Double check that this class is not being created outside of .get()
        if not Config.__is_creating:
            raise Exception("This class is a singleton! Use Config.default() or Config.get() for instances")

    def load(self, filename: str):
        # Read the json file and store as
        raise NotImplementedError("load() has not been implemented yet.")

    def save(self, filename: str):
        # Read the json file and store as
        with open(filename, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=3, sort_keys=True)

    def to_string(self):
        # pp = pprint.PrettyPrinter(indent=2, width=80)

        # return pp.pformat(dataclasses.asdict(self))

        # Using JSON serializer for now since its easier to read. pprint is more compact
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)
