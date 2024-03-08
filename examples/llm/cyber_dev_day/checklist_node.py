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

import ast
import logging
import re
import time
from textwrap import dedent

import openai
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.vectorstores.faiss import FAISS

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm import LLMLambdaNode
from morpheus.llm import LLMNode
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.services.llm_service import LLMService
from morpheus.llm.services.openai_chat_service import OpenAIChatService
from morpheus.llm.services.utils.langchain_llm_client_wrapper import LangchainLLMClientWrapper
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes

from .config import EngineAgentConfig
from .config import EngineChecklistConfig
from .config import EngineCodeRepoConfig
from .config import EngineConfig
from .config import EngineSBOMConfig
from .config import NeMoLLMModelConfig
from .config import NeMoLLMServiceConfig
from .config import NVFoundationLLMModelConfig
from .config import NVFoundationLLMServiceConfig
from .tools import SBOMChecker

logger = logging.getLogger(__name__)

checklist_prompt_template = dedent("""
This is an example of CVE information and a checklist produced to determine if the given CVE is exploitable in a containerized environment:
(1) CVE information: The email module of Python through 3.11.3 incorrectly parses e-mail addresses that contain a special character. The wrong portion of an RFC2822 header is identified as the value of the addr-spec. In some applications, an attacker can bypass a protection mechanism in which application access is granted only after verifying receipt of e-mail to a specific domain (e.g., only @company.example.com addresses may be used for signup). This occurs in email/_parseaddr.py in recent versions of Python.
(2) Checklist:
    1. Check the version of python. The vulnerability affects python through 3.11.3.
    2. Check if the code base uses email functionality in python.

Given the following cve information, make a checklist for security analysts to follow to determine whether a Docker container is vulnerable to this exploit.
CVE information: {{cve_info}}

Checklist:

""").strip("\n")

parselist_prompt_template = dedent("""
Parse the following numbered checklist's contents into a python list in the format ['x', 'y', 'z'], a comma separated list surrounded by square braces. For example, the following checklist:

1. Check for notable vulnerable software vendors
2. Consider the network exposure of your Docker container

Should generate: ["Check for notable vulnerable software vendors", "Consider the network exposure of your Docker container"]

Checklist:
{{template}}""").strip("\n")

# Find all substrings that start and end with quotes, allowing for spaces before a comma or closing bracket
re_quote_capture = re.compile(
    r"""
        (['"])                    # Opening quote
        (                         # Start capturing the quoted content
            (?:\\.|[^\\])*?       # Non-greedy match for any escaped character or non-backslash character
        )                         # End capturing the quoted content
        \1                        # Matching closing quote
        (?=\s*,|\s*\])            # Lookahead for whitespace followed by a comma or closing bracket, without including it in the match
    """,
    re.VERBOSE)


def attempt_fix_list_string(s: str) -> str:
    """
    Attempt to fix unescaped quotes in a string that represents a list to make it parsable.

    Parameters
    ----------
    s : str
        A string representation of a list that potentially contains unescaped quotes.

    Returns
    -------
    str
        The corrected string where internal quotes are properly escaped, ensuring it can be parsed as a list.

    Notes
    -----
    This function is useful for preparing strings to be parsed by `ast.literal_eval` by ensuring that quotes inside
    the string elements of the list are properly escaped. It adds brackets at the beginning and end if they are missing.
    """
    # Check if the input starts with '[' and ends with ']'
    s = s.strip()
    if (not s.startswith('[')):
        s = "[" + s
    if (not s.endswith(']')):
        s = s + "]"

    def fix_quotes(match):
        # Extract the captured groups
        quote_char, content = match.group(1), match.group(2)
        # Escape quotes inside the string content
        fixed_content = re.sub(r"(?<!\\)(%s)" % re.escape(quote_char), r'\\\1', content)
        # Reconstruct the string with escaped quotes and the same quote type as the delimiters
        return f"{quote_char}{fixed_content}{quote_char}"

    # Fix the quotes inside the strings
    fixed_s = re_quote_capture.sub(fix_quotes, s)

    return fixed_s


async def _parse_list(text: list[str]) -> list[list[str]]:
    """
    Asynchronously parse a list of strings, each representing a list, into a list of lists.

    Parameters
    ----------
    text : list of str
        A list of strings, each intended to be parsed into a list.

    Returns
    -------
    list of lists of str
        A list of lists, parsed from the input strings.

    Raises
    ------
    ValueError
        If the string cannot be parsed into a list or if the parsed object is not a list.

    Notes
    -----
    This function tries to fix strings that represent lists with unescaped quotes by calling
    `attempt_fix_list_string` and then uses `ast.literal_eval` for safe parsing of the string into a list.
    It ensures that each element of the parsed list is actually a list and will raise an error if not.
    """
    return_val = []

    for x in text:
        try:
            # Try to do some very basic string cleanup to fix unescaped quotes
            x = attempt_fix_list_string(x)

            # Only proceed if the input is a valid Python literal
            # This isn't really dangerous, literal_eval only evaluates a small subset of python
            current = ast.literal_eval(x)

            # Ensure that the parsed data is a list
            if not isinstance(current, list):
                raise ValueError(f"Input is not a list: {x}")

            # Process the list items
            for i in range(len(current)):
                if (isinstance(current[i], list) and len(current[i]) == 1):
                    current[i] = current[i][0]

            return_val.append(current)
        except (ValueError, SyntaxError) as e:
            # Handle the error, log it, or re-raise it with additional context
            raise ValueError(f"Failed to parse input {x}: {e}")

    return return_val


class CVEChecklistNode(LLMNode):
    """
    A node that orchestrates the process of generating a checklist for CVE (Common Vulnerabilities and Exposures) items.
    It integrates various nodes that handle CVE lookup, prompting, generation, and parsing to produce an actionable checklist.
    """

    def __init__(self, *, config: EngineChecklistConfig):
        """
        Initialize the CVEChecklistNode with optional caching and a vulnerability endpoint retriever.

        Parameters
        ----------
        model_name : str, optional
            The name of the language model to be used for generating text, by default "gpt-3.5-turbo".
        cache_dir : str, optional
            The directory where the node's cache should be stored. If None, caching is not used.
        vuln_endpoint_retriever : object, optional
            An instance of a vulnerability endpoint retriever. If None, defaults to `NISTCVERetriever`.
        """
        super().__init__()

        self._config = config

        llm_service = LLMService.create(config.model.service.type, **config.model.service.model_dump(exclude={"type"}))

        # Add a node to create a prompt for CVE checklist generation based on the CVE details obtained from the lookup
        # node
        self.add_node("checklist_prompt",
                      inputs=[("*", "*")],
                      node=PromptTemplateNode(template=checklist_prompt_template, template_format="jinja"))

        # Instantiate a chat service and configure a client for generating responses to the checklist prompt
        llm_client_1 = llm_service.get_client(**config.model.model_dump(exclude={"service"}))
        self.add_node("generate_checklist", inputs=["/checklist_prompt"], node=LLMGenerateNode(llm_client=llm_client_1))

        # Add a node to parse the generated response into a format suitable for a secondary checklist prompt
        self.add_node("parse_checklist_prompt",
                      inputs=["/generate_checklist"],
                      node=PromptTemplateNode(template=parselist_prompt_template, template_format="jinja"))

        # Configure a second client for generating a follow-up response based on the parsed checklist prompt
        llm_client_2 = llm_service.get_client(**config.model.model_dump(exclude={"service"}))
        self.add_node("parse_checklist",
                      inputs=[("/parse_checklist_prompt", "prompt")],
                      node=LLMGenerateNode(llm_client=llm_client_2))

        # Add an output parser node to process the final generated checklist into a structured list
        self.add_node("output_parser", inputs=["/parse_checklist"], node=LLMLambdaNode(_parse_list), is_output=True)
