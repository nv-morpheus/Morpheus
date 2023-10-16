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
import os
import threading
import typing

import mrc
import mrc.core.operators as ops
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf.errors import PdfStreamError

import cudf

import morpheus._lib.llm as _llm
from morpheus.config import Config
from morpheus.llm import LLMContext
from morpheus.llm import LLMEngine
from morpheus.llm import LLMNodeBase
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class ExtracterNode(LLMNodeBase):

    def __init__(self) -> None:
        super().__init__()

    def get_input_names(self) -> list[str]:
        return []

    async def execute(self, context: LLMContext):

        # Get the keys from the task
        input_keys: list[str] = typing.cast(list[str], context.task()["input_keys"])

        with context.message().payload().mutable_dataframe() as df:
            input_dict: list[dict] = df[input_keys].to_dict(orient="list")

        if (len(input_keys) == 1):
            # Extract just the first key if there is only 1
            context.set_output(input_dict[input_keys[0]])
        else:
            context.set_output(input_dict)

        return context
