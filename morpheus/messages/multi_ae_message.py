# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import dataclasses
import glob
import logging
import typing
from functools import partial

import cupy as cp
import dill
import neo
import numpy as np
import pandas as pd
import torch
from dfencoder import AutoEncoder
from neo.core import operators as ops

from morpheus.config import Config
from morpheus.messages import InferenceMemoryAE
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.messages import UserMessageMeta
from morpheus.pipeline.file_types import FileTypes
from morpheus.pipeline.pipeline import MultiMessageStage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.stages.inference.auto_encoder_inference_stage import MultiInferenceAEMessage
from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MultiAEMessage(MultiMessage):

    model: AutoEncoder

    def get_slice(self, start, stop):
        """
        Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
        and `mess_count`.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.

        Returns
        -------
        morpheus.pipeline.preprocess.autoencoder.MultiAEMessage
            A new `MultiAEMessage` with sliced offset and count.

        """
        return MultiAEMessage(meta=self.meta, mess_offset=start, mess_count=stop - start, model=self.model)
