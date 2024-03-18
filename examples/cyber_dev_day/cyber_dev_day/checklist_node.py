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
from textwrap import dedent

from morpheus.llm import LLMLambdaNode
from morpheus.llm import LLMNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.services.llm_service import LLMService

from .config import EngineChecklistConfig

logger = logging.getLogger(__name__)

# checklist_prompt_template = dedent("""
# This is an example of CVE information and a checklist produced to determine if the given CVE is exploitable in a containerized environment:
# (1) CVE information: The email module of Python through 3.11.3 incorrectly parses e-mail addresses that contain a special character. The wrong portion of an RFC2822 header is identified as the value of the addr-spec. In some applications, an attacker can bypass a protection mechanism in which application access is granted only after verifying receipt of e-mail to a specific domain (e.g., only @company.example.com addresses may be used for signup). This occurs in email/_parseaddr.py in recent versions of Python.
# (2) Checklist:
#     1. Check the version of python. The vulnerability affects python through 3.11.3.
#     2. Check if the code base uses email functionality in python.

# Given the following cve information, make a checklist for security analysts to follow to determine whether a Docker container is vulnerable to this exploit.
# CVE information: {{cve_info}}

# Checklist:

# """).strip("\n")

checklist_prompt_template = dedent(
    """You are an expert security analyst. Your objective is to add a "Checklist" section containing steps to use when assessing the exploitability of a specific CVE within a containerized environment. \
For each checklist item, start with an action verb, making it clear and actionable

**Context**:
Not all CVEs are exploitable in a given container. By making a checklist specific to the information available for a given CVE analysts can execute the checklist to determine exploitability.

**Example Format**:
Below is a format for examples that illustrate transforming CVE information into an exploitability assessment checklist.

Example 1 CVE Details:
- CVE ID: CVE-2022-2309
- Description: NULL Pointer Dereference allows attackers to cause a denial of service (or application crash). This only applies when lxml up to version 4.9.1 \
is used together with libxml2 2.9.10 through 2.9.14. libxml2 2.9.9 and earlier are not affected. It allows triggering crashes through forged input data, given a \
vulnerable code sequence in the application. The vulnerability is caused by the iterwalk function (also used by the canonicalize function). Such code shouldn't be \
in wide-spread use, given that parsing + iterwalk would usually be replaced with the more efficient iterparse function. However, an XML converter that serialises to \
C14N would also be vulnerable, for example, and there are legitimate use cases for this code sequence. If untrusted input is received (also remotely) and processed via \
iterwalk function, a crash can be triggered.
- Vulnerable Package Name: lxml, libxml2
- Vulnerable Package Version: lxml: up to 4.9.1, libxml2: 2.91.0 through 2.9.14
- CVSS3 Vector String: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H

Example 1 Exploitability Assessment Checklist:
[
"Check for lxml: Verify if your project uses the lxml library, which is the affected package. If lxml is not a dependency in your project, then your code is not vulnerable to this CVE.",
"Review Affected Versions: If lxml is used, checked the version that your project depends on. According to the vulnerability details, versions 4.9.0 and earlier are vulnerable.",
"Review Versions of Connected Dependencies: The package is only vulnerable if libxml 2.9.10 through 2.9.14 is also present. Check the version of libxml in the project.",
"Check for use of vulnerable functions: The library is vulnerable through its `iterwalk` function, which is also utilized by the `canonicalize` function. Check if either of these functions are used in your code base."
]

Example 2 CVE Details:
- CVE ID: CVE-2024-23334
- Description: aiohttp is an asynchronous HTTP client/server framework for asyncio and Python. When using aiohttp as a web server and configuring static routes, \
it is necessary to specify the root path for static files. Additionally, the option 'follow_symlinks' can be used to determine whether to follow symbolic links \
outside the static root directory. When 'follow_symlinks' is set to True, there is no validation to check if reading a file is within the root directory. This can \
lead to directory traversal vulnerabilities, resulting in unauthorized access to arbitrary files on the system, even when symlinks are not present. \
Disabling `follow_symlinks` by setting `follow_symlinks = False` and using a reverse proxy are encouraged mitigations. Version 3.9.2 fixes this issue.
- Vulnerable Package Name: aiohttp
- Vulnerable Package Version: from 1.0.5 up to (excluding) 3.9.2
- CVSS3 Vector String: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N

Example 2 Exploitability Assessment Checklist:
[
    "Check for aiohttp: Verify if your project uses the aiohttp library, which is the affected package. If aiohttp is not a dependency in your project, then your code is not vulnerable to this CVE.",
    "Review Affected Versions: If aiohttp is used, check the version that your project depends on. According to the vulnerability details, versions from 1.0.5 up to (excluding) 3.9.2 are affected by this vulnerability.",
    "Review Code To Check for Vulnerability Mitigation: Check if the 'follow_symlinks' option is set to False to mitigate the risk of directory traversal vulnerabilities."
]

**Criteria**:
- Exploitability assessment checklists must relate to the information in the specific CVE Details.
- Exploitability assessment checklists must include checks for mitigating conditions when present in the CVE Details.

**Procedure**:
[
"Understand the CVE Details, description, and CVSS3 attack vector string.",
"Produce a CVE exploitability assessment checklist.",
"Format the checklist as comma separated list surrounded by square braces.",
"Output the checklist."
]

**CVE Details:**
{{cve_details}}

**Checklist**: """).strip("\n")

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

        # Add an output parser node to process the final generated checklist into a structured list
        self.add_node("output_parser", inputs=["/generate_checklist"], node=LLMLambdaNode(_parse_list), is_output=True)