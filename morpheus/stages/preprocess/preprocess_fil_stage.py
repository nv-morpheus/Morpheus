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

import logging
import typing
from functools import partial

import cupy as cp
import neo
import numpy as np
import pandas as pd

import cudf

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage

logger = logging.getLogger(__name__)


class PreprocessFILStage(PreprocessBaseStage):
    """
    FIL usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length
        self.features = c.fil.feature_columns

        assert self._fea_length == len(self.features), \
            f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

    @property
    def name(self) -> str:
        return "preprocess-fil"

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, fea_cols: typing.List[str]) -> MultiInferenceFILMessage:
        """
        For FIL category usecases, this function performs pre-processing.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiMessage`
            Input rows received from Deserialized stage.
        fea_len : int
            Number features are being used in the inference.
        fea_cols : typing.Tuple[str]
            List of columns that are used as features.

        Returns
        -------
        `morpheus.pipeline.messages.MultiInferenceFILMessage`
            FIL inference message.

        """

        try:
            df = x.get_meta(fea_cols)
        except KeyError:
            logger.exception("Cound not get metadat for columns.")
            return None

        # Extract just the numbers from each feature col. Not great to operate on x.meta.df here but the operations will
        # only happen once.
        for col in fea_cols:
            if (df[col].dtype == np.dtype(str) or df[col].dtype == np.dtype(object)):
                # If the column is a string, parse the number
                df[col] = df[col].str.extract(r"(\d+)", expand=False).astype("float32")
            elif (df[col].dtype != np.float32):
                # Convert to float32
                df[col] = df[col].astype("float32")

        if (isinstance(df, pd.DataFrame)):
            df = cudf.from_pandas(df)

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(df.as_gpu_matrix(order='C'))

        count = data.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryFIL(count=count, input__0=data, seq_ids=seg_ids)

        infer_message = MultiInferenceFILMessage(meta=x.meta,
                                                 mess_offset=x.mess_offset,
                                                 mess_count=x.mess_count,
                                                 memory=memory,
                                                 offset=0,
                                                 count=memory.count)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        return partial(PreprocessFILStage.pre_process_batch, fea_len=self._fea_length, fea_cols=self.features)

    def _get_preprocess_node(self, seg: neo.Builder):
        return neos.PreprocessFILStage(seg, self.unique_name, self.features)
