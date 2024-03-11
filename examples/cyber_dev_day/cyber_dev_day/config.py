import typing

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag


def _llm_discriminator(v: typing.Any) -> str:

    if isinstance(v, dict):
        return v.get("service").get("type")
    return getattr(getattr(v, "service"), "type")


class NeMoLLMServiceConfig(BaseModel):

    type: typing.Literal["nemo"] = "nemo"

    api_key: str | None = None
    org_id: str | None = None


class NeMoLLMModelConfig(BaseModel):

    service: NeMoLLMServiceConfig

    model_name: str
    customization_id: str | None = None
    temperature: float = 0.0
    tokens_to_generate: int = 300


class NVFoundationLLMServiceConfig(BaseModel):

    type: typing.Literal["nvfoundation"] = "nvfoundation"

    api_key: str | None = None


class NVFoundationLLMModelConfig(BaseModel):

    service: NVFoundationLLMServiceConfig

    model_name: str
    temperature: float = 0.0


class OpenAIServiceConfig(BaseModel):

    type: typing.Literal["openai"] = "openai"


class OpenAIMModelConfig(BaseModel):

    service: OpenAIServiceConfig

    model_name: str


LLMModelConfig = typing.Annotated[typing.Annotated[NeMoLLMModelConfig, Tag("nemo")]
                                  | typing.Annotated[OpenAIMModelConfig, Tag("openai")]
                                  | typing.Annotated[NVFoundationLLMModelConfig, Tag("nvfoundation")],
                                  Discriminator(_llm_discriminator)]


class HttpServerInputConfig(BaseModel):

    type: typing.Literal["http_server"] = "http_server"


class NspectFileInputConfig(BaseModel):

    type: typing.Literal["nspect_file"] = "nspect_file"


class CveFileInputConfig(BaseModel):

    type: typing.Literal["cve_file"] = "cve_file"


class EngineChecklistConfig(BaseModel):

    model: LLMModelConfig


class EngineSBOMConfig(BaseModel):

    data_file: str


class EngineCodeRepoConfig(BaseModel):

    faiss_dir: str

    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"


class EngineAgentConfig(BaseModel):

    model: LLMModelConfig

    sbom: EngineSBOMConfig

    code_repo: EngineCodeRepoConfig


class EngineConfig(BaseModel):

    checklist: EngineChecklistConfig

    agent: EngineAgentConfig
