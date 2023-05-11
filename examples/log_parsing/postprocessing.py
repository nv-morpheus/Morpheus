# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import json
import pathlib
import typing
from collections import defaultdict

import mrc
import numpy as np
import pandas as pd
from mrc.core import operators as ops

from messages import MultiPostprocLogParsingMessage
from messages import MultiResponseLogParsingMessage
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("log-postprocess", modes=[PipelineModes.NLP])
class LogParsingPostProcessingStage(SinglePortStage):

    def __init__(self, c: Config, vocab_path: pathlib.Path, model_config_path: pathlib.Path):
        """
        Post-processing stage for log parsing pipeline.

        Parameters
        ----------
        c : `morpheus.config.Config`
            The morpheus config
        vocab_path : pathlib.Path, exists = True, dir_okay = False
            Model vocab file to use for post-processing
        model_config_path : pathlib.Path, exists = True, dir_okay = False
            Model config file
        """
        super().__init__(c)

        self._vocab_path = vocab_path
        self._model_config_path = model_config_path

        self._vocab_lookup = {}

        # Explicitly setting the encoding, we know we have unicode chars in this file and we need to avoid issue:
        # https://github.com/nv-morpheus/Morpheus/issues/859
        with open(vocab_path, encoding='UTF-8') as f:
            for index, line in enumerate(f):
                self._vocab_lookup[index] = line.split()[0]

        with open(model_config_path, encoding='UTF-8') as f:
            config = json.load(f)

        self._label_map = {int(k): v for k, v in config["id2label"].items()}

    @property
    def name(self) -> str:
        return "logparsing-postproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiResponseLogParsingMessage, )

    def _postprocess(self, x: MultiPostprocLogParsingMessage):

        infer_pdf = pd.DataFrame(x.seq_ids.get()).astype(int)
        infer_pdf.columns = ["doc", "start", "stop"]
        infer_pdf["confidences"] = x.confidences.tolist()
        infer_pdf["labels"] = x.labels.tolist()
        infer_pdf["token_ids"] = x.input_ids.tolist()

        infer_pdf["confidences"] = infer_pdf.apply(lambda row: row["confidences"][row["start"]:row["stop"]], axis=1)

        infer_pdf["labels"] = infer_pdf.apply(lambda row: row["labels"][row["start"]:row["stop"]], axis=1)

        infer_pdf["token_ids"] = infer_pdf.apply(lambda row: row["token_ids"][row["start"]:row["stop"]], axis=1)

        # aggregated logs
        infer_pdf = infer_pdf.groupby("doc").agg({"token_ids": "sum", "confidences": "sum", "labels": "sum"})

        # parse_by_label
        parsed_dfs = infer_pdf.apply(lambda row: self.__get_label_dicts(row), axis=1, result_type="expand")

        ext_parsed = pd.DataFrame(parsed_dfs[0].tolist())
        ext_confidence = pd.DataFrame(parsed_dfs[1].tolist())
        parsed_df = pd.DataFrame()
        confidence_df = pd.DataFrame()
        ext_confidence = ext_confidence.applymap(np.mean)
        for label in ext_parsed.columns:
            if label[0] == "B":
                col_name = label[2:]
                if "I-" + col_name in ext_parsed.columns:
                    parsed_df[col_name] = ext_parsed[label] + " " + ext_parsed["I-" + col_name].fillna('')
                    confidence_df[col_name] = (ext_confidence[label] + ext_confidence[label]) / 2
                else:
                    parsed_df[col_name] = ext_parsed[label]
                    confidence_df[col_name] = ext_confidence[label]

        # decode cleanup
        parsed_df = self.__decode_cleanup(parsed_df)

        return MessageMeta(df=parsed_df)

    def __get_label_dicts(self, row):
        token_dict = defaultdict(str)
        confidence_dict = defaultdict(list)
        for label, confidence, token_id in zip(row["labels"], row["confidences"], row["token_ids"]):
            text_token = self._vocab_lookup[token_id]
            if text_token[:2] != "##" and text_token[0] != '.':
                # if not a subword use the current label, else use previous
                new_label = label
                new_confidence = confidence
            if self._label_map[new_label] in token_dict:
                token_dict[self._label_map[new_label]] = (token_dict[self._label_map[new_label]] + " " + text_token)
            else:
                token_dict[self._label_map[new_label]] = text_token
            confidence_dict[self._label_map[label]].append(new_confidence)
        return token_dict, confidence_dict

    def __decode_cleanup(self, df):
        df.replace(r"\s+##", "", regex=True, inplace=True)
        df.replace(r"\s+\.+\s", ".", regex=True, inplace=True)
        df.replace(r"\s+:+\s", ":", regex=True, inplace=True)
        df.replace(r"\s+\|+\s", "|", regex=True, inplace=True)
        df.replace(r"\s+\++\s", "+", regex=True, inplace=True)
        df.replace(r"\s+\-+\s", "-", regex=True, inplace=True)
        df.replace(r"\s+\<", "<", regex=True, inplace=True)
        df.replace(r"\<+\s", "<", regex=True, inplace=True)
        df.replace(r"\s+\>", ">", regex=True, inplace=True)
        df.replace(r"\>+\s", ">", regex=True, inplace=True)
        df.replace(r"\s+\=+\s", "=", regex=True, inplace=True)
        df.replace(r"\s+\#+\s", "#", regex=True, inplace=True)
        df.replace(r"\[+\s", "[", regex=True, inplace=True)
        df.replace(r"\s\]", "]", regex=True, inplace=True)
        df.replace(r"\(+\s", "(", regex=True, inplace=True)
        df.replace(r"\s\)", ")", regex=True, inplace=True)
        df.replace(r"\s\"", "\"", regex=True, inplace=True)
        df.replace(r"\"+\s", "\"", regex=True, inplace=True)
        df.replace(r"\\+\s", "\"", regex=True, inplace=True)
        df.replace(r"\s+_+\s", "_", regex=True, inplace=True)
        df.replace(r"\s+/", "/", regex=True, inplace=True)
        df.replace(r"/+\s", "/", regex=True, inplace=True)
        df.replace(r"\s+\?+\s", "?", regex=True, inplace=True)
        df.replace(r"\s+;+\s", "; ", regex=True, inplace=True)

        return df

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        stream = builder.make_node(self.unique_name, ops.map(self._postprocess))

        builder.make_edge(input_stream[0], stream)

        return stream, MessageMeta
