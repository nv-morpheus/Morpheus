# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
import pytest

from _utils import assert_results
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm.llm_engine_stage import LLMEngineStage
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

MULTI_LINE_JINJA_TEMPLATE = """Testing a loop:
{% for lv in ctx.list_values -%}Title: {{ lv.title }}, Summary: {{ lv.summary }}
{% endfor %}
{{ ctx.query }}"""


def _build_engine(template: str, template_format: str) -> LLMEngine:
    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_node("prompts",
                    inputs=["/extracter"],
                    node=PromptTemplateNode(template=template, template_format=template_format))

    engine.add_task_handler(inputs=["/prompts"], handler=SimpleTaskHandler())

    return engine


@pytest.mark.use_python
@pytest.mark.parametrize(
    "template,template_format,values,expected_output",
    [
        ("Hello {name}!",
         "f-string", {
             'name': ['World', 'Universe', 'Galaxy', 'Moon']
         }, ["Hello World!", "Hello Universe!", "Hello Galaxy!", "Hello Moon!"]),
        ("Hello {{ name }}!",
         "jinja", {
             'name': ['World', 'Universe', 'Galaxy', 'Moon']
         }, ["Hello World!", "Hello Universe!", "Hello Galaxy!", "Hello Moon!"]),
        #  ("I would like one {fruit} and one {vegetable}.",
        #   "f-string", {
        #       'fruit': ['apple', 'plum'], 'vegetable': ['carrot', 'broccoli']
        #   }, ["I would like one apple and one carrot.", "I would like one plum and one broccoli."]),
        # ("I would like one {{ fruit }} and one {{ vegetable }}.",
        #  "jinja", {
        #      'fruit': ['apple', 'plum'], 'vegetable': ['carrot', 'broccoli']
        #  }, ["I would like one apple and one carrot.", "I would like one plum and one broccoli."]),
        # (MULTI_LINE_JINJA_TEMPLATE,
        #  "jinja",
        #  {
        #      'list_values': [[{
        #          'title': 'title1', 'summary': 'summary1'
        #      }, {
        #          'title': 'title2', 'summary': 'summary2'
        #      }], [{
        #          'title': 'rockets', 'summary': 'space'
        #      }]],
        #      'query': ['query1', 'query2']
        #  },
        #  [
        #      "Testing a loop:\nTitle: title1, Summary: summary1\nTitle: title2, Summary: summary2\n\nquery1",
        #      "Testing a loop:\nTitle: rockets, Summary: space\n\nquery2",
        #  ])
    ],
    ids=["f-string-hello-world", "jinja-hello-world"])
def test_prompt_template_node_pipe(config: Config,
                                   template: str,
                                   template_format: str,
                                   values: dict,
                                   expected_output: list[str]):
    config.mode = PipelineModes.OTHER
    input_df = cudf.DataFrame(values)
    expected_df = input_df.copy(deep=True)
    expected_df["response"] = expected_output

    input_names = sorted(values.keys())
    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": input_names}}

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(template=template, template_format=template_format)))
    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    assert_results(sink.get_results())
