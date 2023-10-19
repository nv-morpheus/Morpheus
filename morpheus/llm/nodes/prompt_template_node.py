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

import asyncio
import logging
import typing

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase

logger = logging.getLogger(__name__)


class PromptTemplateNode(LLMNodeBase):

    def __init__(self, template: str, template_format: typing.Literal["f-string", "jinja"] = "f-string") -> None:
        super().__init__()

        self._input_variables = ["question"]
        self._template = template
        self._template_format = template_format

        if (self._template_format == "f-string"):
            self._input_names = []
        elif (self._template_format == "jinja"):
            from jinja2 import Template
            from jinja2 import meta

            jinja_template = Template(self._template)

            self._input_names = list(meta.find_undeclared_variables(jinja_template.environment.parse(self._template)))

    def get_input_names(self):
        return self._input_names

    async def execute(self, context: LLMContext):

        # if ("0" in context.input_map[0].input_name):
        #     context.parent.get_input("$.chat1.choices[*][0].message.content")

        # Get the keys from the task
        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        if (self._template_format == "f-string"):
            output_list = [self._template.format(**x) for x in input_list]
        elif (self._template_format == "jinja"):

            from jinja2 import Template

            template = Template(self._template, enable_async=True)

            render_coros = [template.render_async(**inputs) for inputs in input_list]

            output_list = await asyncio.gather(*render_coros)

        context.set_output(output_list)

        return context
