# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

import srf

<<<<<<< HEAD
#import cuml
from xgboost import XGBClassifier
=======
import cuml

>>>>>>> branch-22.08
from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .graph_sage_stage import GraphSAGEMultiMessage
import cupy as cp

class ClassificationStage(SinglePortStage):

    def __init__(self, c: Config, model_xgb_file: str):
        super().__init__(c)

        #self._xgb_model = cuml.ForestInference.load(model_xgb_file, output_class=True)
        self._xgb_model = XGBClassifier()
        self._xgb_model.load_model(model_xgb_file)

    @property
    def name(self) -> str:
        return "gnn-fraud-classification"

    def accepted_types(self) -> typing.Tuple:
        return (GraphSAGEMultiMessage, )

    def _process_message(self, message: GraphSAGEMultiMessage):
        ind_emb_columns = message.get_meta(message.inductive_embedding_column_names)

        message.set_meta("node_id", message.node_identifiers)
      #  import IPython; IPython.embed(); exit(1)
        #ind_emb_columns = cp.random.randn(265, 178)
        #prediction = self._xgb_model.predict_proba(ind_emb_columns).iloc[:, 1]
        prediction = self._xgb_model.predict_proba(ind_emb_columns)[:, 1]

        message.set_meta("prediction", prediction)

        return message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self._process_message)
        builder.make_edge(input_stream[0], node)
        return node, MultiMessage
    
    def supports_cpp_node(self):
        # Get the value from the worker class
        return False

