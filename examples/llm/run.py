from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.messages import MessageMeta, MultiMessage

from morpheus.pipeline import LinearPipeline
from morpheus.config import Config

import cudf
import mrc
import typing
import asyncio
from abc import ABC, abstractmethod
import nemollm
import openai

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
        return f"{prompt} {self._response}"

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
        return f"{prompt}{response}"

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
        return f"{prompt}{result.choices[0].text}"

    async def generate_multiple_async(self, prompts: [str]) -> [str]:
      tasks = [asyncio.ensure_future(self.generate_async(prompt)) for prompt in prompts]
      await asyncio.wait(tasks)
      return [task.result() for task in tasks]

class MyFlatmapStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "flatmap"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MultiMessage):
        new_series = message.get_meta("value")
        new_df = cudf.DataFrame({"value": new_series})
        new_meta = MessageMeta(new_df)
        new_message = MultiMessage(meta=new_meta)
        return new_message

    async def on_data_async_gen(self, message: MultiMessage):
        for duration in message.get_meta("sleep").to_pandas():
            await asyncio.sleep(duration)
            new_series = message.get_meta("value")
            new_df = cudf.DataFrame({"value": new_series})
            new_meta = MessageMeta(new_df)
            new_message = MultiMessage(meta=new_meta)
            yield new_message

    def _build_single(self, builder: mrc.Builder, input: StreamPair) -> StreamPair:
        [input_node, input_type] = input
        # node = builder.make_node(self.unique_name, mrc.operators.map(self.on_data))
        node = builder.make_node(self.unique_name, mrc.operators.concat_map_async(self.on_data_async_gen))
        builder.make_edge(input_node, node)
        return node, input_type

def test_nemo_pipeline():

    input_df_a = cudf.DataFrame({ "sleep": [1], "value": [0] })
    input_df_b = cudf.DataFrame({ "sleep": [2], "value": [1] })
    input_df_c = cudf.DataFrame({ "sleep": [1], "value": [2] })
    config = Config()
    pipeline = LinearPipeline(config)

    pipeline.set_source(InMemorySourceStage(config, [input_df_a, input_df_b, input_df_c]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MyFlatmapStage(config))
    pipeline.add_stage(SerializeStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    messages: list[MessageMeta] = sink.get_messages()

    for message in messages:
        print(message.copy_dataframe())

async def run_client():
    import os

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
    main()