from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.messages import MessageMeta, MultiMessage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.utils.logger import configure_logging

from morpheus.pipeline import LinearPipeline
from morpheus.config import Config

import logging
import cudf
import mrc
import typing
import asyncio
from abc import ABC, abstractmethod
import nemollm
import openai
import os

class LLMClient(ABC):

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_multiple_async(self, prompts: [str]) -> [str]:
        pass

class MockLLMClient(LLMClient):

    _response: str

    def __init__(self, response: str):
        self._response = response

    def _query(self, prompt: str) -> str:
        return self._response

    async def generate_async(self, prompt: str) -> str:
        return self._query(prompt)

    async def generate_multiple_async(self, prompts: [str]) -> [str]:
        return [self._query(prompt) for prompt in prompts]

class NemoLLMClient(LLMClient):

    _connection: nemollm.NemoLLM
    _model: str

    def __init__(
            self,
            api_key: str | None = None,
            org_id: str | None = None,
            api_host: str | None = None,
            model: str | None = None
    ):
        self._connection = nemollm.NemoLLM(api_key, org_id, api_host)
        self._model = model or "gpt5b"

    async def generate_async(self, prompt: str) -> str:
        future = self._connection.generate(self._model, prompt, return_type="async")
        result = await asyncio.wrap_future(future)
        response = nemollm.NemoLLM.post_process_generate_response(result, return_text_completion_only=True)
        return response

    async def generate_multiple_async(self, prompts: [str]) -> [str]:
      tasks = [asyncio.ensure_future(self.generate_async(prompt)) for prompt in prompts]
      await asyncio.wait(tasks)
      return [task.result() for task in tasks]

class OpenAICompletionLLMClient(LLMClient):

    _api_key: str
    _model: str

    def __init__(
            self,
            api_key: str | None = None,
            model: str | None = None
    ):
        self._api_key = api_key
        self._model = model or "ada"

    async def generate_async(self, prompt: str) -> str:
        result = await openai.Completion.acreate(model=self._model, prompt=prompt)
        return result.choices[0].text

    async def generate_multiple_async(self, prompts: [str]) -> [str]:
      tasks = [asyncio.ensure_future(self.generate_async(prompt)) for prompt in prompts]
      await asyncio.wait(tasks)
      return [task.result() for task in tasks]

class PromptCompletionStage(SinglePortStage):

    _client: LLMClient
    _prompt_key: str
    _response_key: str

    def __init__(
            self,
            config: Config,
            client: LLMClient,
            prompt_key:str="prompt",
            response_key:str="response"
    ):
        super().__init__(config)
        self._client = client
        self._prompt_key = prompt_key
        self._response_key = response_key

    @property
    def name(self) -> str:
        return "flatmap"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    async def on_data_async_gen(self, message: MultiMessage):
        prompts = [prompt for prompt in message.get_meta(self._prompt_key).to_pandas()]
        responses = await self._client.generate_multiple_async(prompts)
        message.set_meta(self._response_key, responses)
        yield message

    def _build_single(self, builder: mrc.Builder, input: StreamPair) -> StreamPair:
        [input_node, input_type] = input
        node = builder.make_node(self.unique_name, mrc.operators.concat_map_async(self.on_data_async_gen))
        builder.make_edge(input_node, node)
        return node, input_type

def test_nemo_pipeline():

    # configure_logging(log_level=logging.DEBUG)

    input_df_a = cudf.DataFrame({"prompt": ["3+5=", "4-2=", "7*1="]})
    input_df_b = cudf.DataFrame({"prompt": ["the man wore a red"]})
    input_df_c = cudf.DataFrame({"prompt": ["blue+red=", "yellow+blue="]})

    config = Config()
    pipeline = LinearPipeline(config)

    # ngc_api_key=os.environ.get("NGC_API_KEY")
    # llm_client=NemoLLMClient(ngc_api_key, model="gpt-43b-002")

    # openai_api_key=os.environ.get("OPENAI_API_KEY")
    # llm_client=OpenAICompletionLLMClient(openai_api_key)

    llm_client=MockLLMClient(response="cool story")

    pipeline.set_source(InMemorySourceStage(config, [input_df_a, input_df_b, input_df_c]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(PromptCompletionStage(config, llm_client))
    pipeline.add_stage(SerializeStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    messages: list[MessageMeta] = sink.get_messages()

    for message in messages:
        print(message.copy_dataframe())

async def run_client():
    # openai_api_key=os.environ.get("OPENAI_API_KEY")
    # client=OpenAICompletionLLMClient(openai_api_key)

    # ngc_api_key=os.environ.get("NGC_API_KEY")
    # client=NemoLLMClient(ngc_api_key)

    client=MockLLMClient(response="cool story")

    responses = await client.generate_multiple_async([
        "the numbers 5 and 6 sum to",
        "blue and red combine to form"
    ])

    print(responses)


def main():
    asyncio.run(run_client())

if __name__ == '__main__':
    test_nemo_pipeline()
    # main()
