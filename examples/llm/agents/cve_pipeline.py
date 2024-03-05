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

from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.vectorstores.faiss import FAISS

import cudf

from morpheus._lib.llm import LLMLambdaNode
from morpheus._lib.llm import LLMNode
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.services.openai_chat_service import OpenAIChatService
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes

from .tools import SBOMChecker

logger = logging.getLogger(__name__)

checklist_prompt_template = """This is an example of (1) CVE information, and (2) a checklist produced to determine if a given CVE is exploitable in a containerized environment:
(1) CVE Information:
CVE Description: DISPUTED: In Apache Batik 1.x before 1.10, when deserializing subclass of `AbstractDocument`, the class takes a string from the inputStream as the class name which then use it to call the no-arg constructor of the class. Fix was to check the class type before calling newInstance in deserialization.
CVSS Vector: CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
CWE Name: CWE-502: Deserialization of Untrusted Data (4.11)
CWE Description: The product deserializes untrusted data without sufficiently verifying that the resulting data will be valid. It is often convenient to serialize objects for communication or to save them for later use. However, deserialized data or code can often be modified without using the provided accessor functions if it does not use cryptography to protect itself. Furthermore, any cryptography would still be client-side security -- which is a dangerous security assumption. Data that is untrusted can not be trusted to be well-formed. When developers place no restrictions on gadget chains, or series of instances and method invocations that can self-execute during the deserialization process (i.e., before the object is returned to the caller), it is sometimes possible for attackers to leverage them to perform unauthorized actions, like generating a shell.
Notable Vulnerable Software Vendors: ['Oracle', 'Apache', 'Canonical', 'Debian']
(2) Checklist:
Based on the information available, CVE-2018-8013 is a critical vulnerability that involves the deserialization of untrusted data in Apache Batik before version 1.10. The Common Vulnerability Scoring System (CVSS) 3.x score is 9.8, indicating a critical severity.
Here's an example of preliminary checklist similiar to what you should create to determine if a Docker container is vulnerable to this exploit:
1. Check the Apache Batik version: The vulnerability affects Apache Batik versions 1.0 to 1.9.1. If your Docker container is running Apache Batik, verify the version. If it's between 1.0 and 1.9.1, it's vulnerable and you should update to version 1.10 or later.
2. Check for any software using vulnerable Apache Batik versions: It's important to note that other applications might use Apache Batik and thus be vulnerable. Check all applications within your Docker container for dependencies on vulnerable versions of Apache Batik. Some notable software includes certain versions of Ubuntu Linux, Debian Linux, and multiple Oracle applications.
3. Evaluate the deserialization risk: Since the vulnerability involves deserialization of untrusted data, you should consider whether your applications are performing such operations. If they are, consider if these deserialization operations are using untrusted, user-supplied data. If so, this is a potential attack vector.
4. Network exposure: The vulnerability has a network attack vector. Consider whether your Docker container has network exposure, especially if the Apache Batik service or any service using it is exposed to the internet.
5. Disputed status: CVE-2018-8013 is disputed and may not be a vulnerability.

Given the following information about {{cve}}, make preliminary checklist for a security analyst to follow to determine whether a Docker container is vulnerable to this exploit.
CVE description: {{cve_description}}
{% if cvss_vector %}
CVSS Vector: {{cvss_vector}}
{% endif %}
{% if cwe_name %}
CWE Name: {{cwe_name}}
{% endif %}
{% if cwe_description %}
CWE Description: {{cwe_description}}
{% endif %}
{% if cwe_extended_description %}
{{cwe_extended_description}}
{% endif %}
{% if vendor_names %}
Notable Vulnerable Software Vendors: {{vendor_names}}
{% endif %}
"""

parselist_prompt_template = """Parse the following numbered checklist into a python list in the format ["x", "y", "z"], a comma separated list surrounded by square braces: {{template}}"""

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

    def __init__(self,
                 *,
                 model_name: str = "gpt-3.5-turbo",
                 cache_dir: str = None,
                 vuln_endpoint_retriever=None,
                 max_retries: int = 10):
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

        chat_service = OpenAIChatService()

        # Add a node to create a prompt for CVE checklist generation based on the CVE details obtained from the lookup
        # node
        self.add_node("checklist_prompt",
                      inputs=[("/lookup/*", "*"), "cve"],
                      node=PromptTemplateNode(template=checklist_prompt_template, template_format="jinja"))

        # Instantiate a chat service and configure a client for generating responses to the checklist prompt
        llm_client_1 = chat_service.get_client(model_name=model_name, temperature=0)
        self.add_node("generate_checklist", inputs=["/checklist_prompt"], node=LLMGenerateNode(llm_client=llm_client_1))

        # Add a node to parse the generated response into a format suitable for a secondary checklist prompt
        self.add_node("parse_checklist_prompt",
                      inputs=["/generate_checklist"],
                      node=PromptTemplateNode(template=parselist_prompt_template, template_format="jinja"))

        # Configure a second client for generating a follow-up response based on the parsed checklist prompt
        llm_client_2 = chat_service.get_client(model_name=model_name, temperature=0)
        self.add_node("parse_checklist",
                      inputs=[("/parse_checklist_prompt", "prompt")],
                      node=LLMGenerateNode(llm_client=llm_client_2))

        # Add an output parser node to process the final generated checklist into a structured list
        self.add_node("output_parser", inputs=["/chat2"], node=LLMLambdaNode(_parse_list), is_output=True)


def _build_agent_executor(model_name: str) -> AgentExecutor:

    llm = OpenAI(model=model_name, temperature=0.0, client=None)

    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools: list[Tool] = []

    if ((sbom_parser is not None)
            and ((sbom_file_path is not None) or ((nspect_id is not None) and (ngc_image_url is not None)))):
        if isinstance(sbom_parser, str):
            assert (sbom_parser in SBOM_PICKER), 'Parser not found'
            sbom_parser = SBOM_PICKER[sbom_parser]
            parser = sbom_parser(file=sbom_file_path, nspect_id=nspect_id, image_url_name=ngc_image_url)
        elif isclass(sbom_parser):
            try:
                class_name = sbom_parser.get_name()
            except:
                class_name = None
            assert (class_name in SBOM_PICKER), 'Parser not found'
            parser = sbom_parser(file=sbom_file_path, nspect_id=nspect_id, image_url_name=ngc_image_url)
        elif isinstance(sbom_parser, SBOMParser):
            assert sbom_parser.get_name() in SBOM_PICKER, 'Parser not found'
            parser = sbom_parser
        else:
            assert None, 'Parser not found'

        sbom_map = parser.parse()
        sbom_checker = SBOMChecker(sbom_map)

        tools.append(
            Tool(name="SBOM Package Checker",
                 func=sbom_checker.sbom_checker,
                 description=("useful for when you need to check the Docker container's software bill of "
                              "materials (SBOM) to get whether or not a given library is in the container. "
                              "Input should be the name of the library or software. If the package is "
                              "present a version number is returned, otherwise False is returned if the "
                              "package is not present.")))

    if (code_faiss_dir is not None):
        # load code vector DB
        code_vector_db = FAISS.load_local(code_faiss_dir, embeddings)
        code_qa_tool = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=code_vector_db.as_retriever())
        tools.append(
            Tool(name="Docker Container Code QA System",
                 func=code_qa_tool.run,
                 description=("useful for when you need to check if an application or any dependency within "
                              "the Docker container uses a function or a component of a library.")))

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_engine(model_name: str) -> LLMEngine:

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node("checklist", inputs=["/extracter/*"], node=CVEChecklistNode(model_name=model_name))

    engine.add_node("agent",
                    inputs=[("/extracter")],
                    node=LangChainAgentNode(agent_executor=_build_agent_executor(model_name=model_name)))

    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    return engine


def pipeline(
    num_threads: int,
    pipeline_batch_size,
    model_max_batch_size,
    model_name,
    repeat_count,
) -> float:
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    source_dfs = [
        cudf.DataFrame(
            {"questions": ["Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"]})
    ]

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs, repeat=repeat_count))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name)))

    sink = pipe.add_stage(InMemorySinkStage(config))

    start_time = time.time()

    pipe.run()

    messages = sink.get_messages()
    responses = concat_dataframes(messages)

    logger.info("Pipeline complete. Received %s responses:\n%s", len(messages), responses['response'])

    return start_time
