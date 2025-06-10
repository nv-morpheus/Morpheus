# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import collections.abc
import logging
import re

import mrc
import pandas as pd
import yaml
from datasets import load_dataset

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.type_utils import exec_mode_to_df_type_str
from morpheus.utils.type_utils import get_df_class

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("datasets-source")
class DatasetsSourceStage(PreallocatorMixin, GpuAndCpuMixin, SingleOutputSource):
    """
    Source stage that loads data from the datasets library into a DataFrame.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    dataset_names : list[str]
        List of dataset names to load from the datasets library.
    num_samples : int | None, optional
        Number of samples to load from each dataset. If not specified, all samples will be loaded.
    repeat : int, default = 1, min = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    include_privacy_masks : bool, optional
        Whether to include privacy masks in the output DataFrame. Defaults to False.
    """

    AVAILABLE_DATASETS = {
        "gretel": "gretelai/gretel-pii-masking-en-v1",
    }

    def __init__(self,
                 config: Config,
                 dataset_names: list[str],
                 num_samples: int | None = None,
                 repeat: int = 1,
                 include_privacy_masks: bool = False):
        super().__init__(config)

        for name in dataset_names:
            if name not in self.AVAILABLE_DATASETS:
                raise ValueError(f"Unknown dataset: {name}")

        self._dataset_names = dataset_names
        self._num_samples = num_samples
        self._df_str = exec_mode_to_df_type_str(config.execution_mode)
        self._df_class = get_df_class(config.execution_mode)
        self._include_privacy_masks = include_privacy_masks
        self._repeat_count = repeat

    @property
    def name(self) -> str:
        return "from-datasets"

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self.source_generator)

    @staticmethod
    def fix_gretel_masks(row) -> list[dict[str, str]]:
        """Fix Gretel dataset mask format to match standard format."""
        list_of_masks = []
        for m in row.privacy_mask:
            label = m["types"][0]
            value = m["entity"]
            matches = list(re.finditer(re.escape(value), row.source_text))
            if len(matches) > 1:
                print("WARNING: found more than one of the entity!")

            match = matches[0]
            list_of_masks.append({
                "label": label,
                "start": match.start(),
                "end": match.end(),
                "value": value,
            })

        return list_of_masks

    @classmethod
    def process_gretel_dataset(cls, df: pd.DataFrame, num_samples: int | None,
                               include_privacy_masks: bool) -> pd.DataFrame:
        """Process Gretel dataset to standard format."""
        source_columns = ["text"]
        output_columns = ["source_text"]
        if include_privacy_masks:
            source_columns.append("entities")
            output_columns.append("privacy_mask")

        df = df[source_columns]
        df.columns = output_columns

        if num_samples is not None:
            df = df.sample(n=num_samples).reset_index(drop=True)

        if include_privacy_masks:
            df["privacy_mask"] = df["privacy_mask"].apply(yaml.safe_load)
            df["privacy_mask"] = df.apply(cls.fix_gretel_masks, axis=1)

        df["source"] = "gretel"

        return df

    @staticmethod
    def normalize_privacy_masks(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize privacy masks to consistent format."""
        for _, row in dataframe.iterrows():
            list_of_masks = []
            for m in row.privacy_mask:
                list_of_masks.append({
                    "label": m["label"],
                    "start": m["start"],
                    "end": m["end"],
                    "value": m["value"],
                })
            row.privacy_mask = list_of_masks

        return dataframe

    def source_generator(self, subscription: mrc.Subscription) -> collections.abc.Iterator[MessageMeta]:

        for dataset_name in self._dataset_names:
            dataset = load_dataset(self.AVAILABLE_DATASETS[dataset_name], split="validation")
            df = dataset.to_pandas()

            if dataset_name == "gretel":
                df = self.process_gretel_dataset(df, self._num_samples, self._include_privacy_masks)

            if self._include_privacy_masks:

                df = self.normalize_privacy_masks(df)
                df = df[df.privacy_mask.str.len() > 0].reset_index(drop=True)

            if self._df_str == "cudf":
                df = self._df_class(df)

            for i in range(self._repeat_count):
                if not subscription.is_subscribed():
                    break

                msg = MessageMeta(df)

                # If we are looping, copy the object. Do this before we push the object in case it changes
                if (i + 1 < self._repeat_count):
                    df = df.copy()

                    # Shift the index to allow for unique indices without reading more data
                    df.index += len(df)

                yield msg
