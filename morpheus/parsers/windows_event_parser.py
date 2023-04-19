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

import logging
import os
import typing

import cudf

from morpheus.parsers.event_parser import EventParser

log = logging.getLogger(__name__)


class WindowsEventParser(EventParser):
    """
    This is class parses windows event logs.

    Parameters
    ----------
    interested_eventcodes: typing.Set[int]
        Set of interested codes to parse
    """
    REGEX_FILE = "resources/windows_event_regex.yaml"
    EVENT_NAME = "windows event"

    def __init__(self, interested_eventcodes=None):
        regex_filepath = (os.path.dirname(os.path.abspath(__file__)) + "/" + self.REGEX_FILE)
        self.interested_eventcodes = interested_eventcodes
        self.event_regex = self._load_regex_yaml(regex_filepath)
        EventParser.__init__(self, self.get_columns(), self.EVENT_NAME)

    def parse(self, text: cudf.Series) -> cudf.Series:
        """Parses the Windows raw event.

        Parameters
        ----------
        text : cudf.Series
            Raw event log text to be parsed

        Returns
        -------
        cudf.DataFrame
            Parsed logs dataframe
        """
        # Clean raw data to be consistent.
        text = self.clean_raw_data(text)
        output_chunks = []
        for eventcode in self.event_regex.keys():
            pattern = "eventcode=%s" % (eventcode)
            # input_chunk = self.filter_by_pattern(dataframe, raw_column, pattern)
            input_chunk = text[text.str.contains(pattern)]
            if not input_chunk.empty:
                temp = self.parse_raw_event(input_chunk, self.event_regex[eventcode])
                if not temp.empty:
                    output_chunks.append(temp)
        parsed_dataframe = cudf.concat(output_chunks)
        # Replace null values with empty.
        parsed_dataframe = parsed_dataframe.fillna("")
        return parsed_dataframe

    def clean_raw_data(self, text: cudf.Series) -> cudf.Series:
        """
        Lower casing and replacing escape characters.

        Parameters
        ----------
        text : cudf.Series
            Raw event log text to be clean
        event_regex: typing.Dict[str, any]
            Required regular expressions for a given event type

        Returns
        -------
        cudf.Series
            Clean raw event log text
        """
        text = (text.str.lower().str.replace("\\\\t", "").str.replace("\\\\r", "").str.replace("\\\\n", "|"))
        return text

    def _load_regex_yaml(self, yaml_file):
        event_regex = EventParser._load_regex_yaml(self, yaml_file)
        if self.interested_eventcodes is not None:
            for eventcode in self.interested_eventcodes:
                required_event_regex = {}
                if eventcode not in event_regex:
                    raise KeyError("Regex for eventcode %s is not available in the config file. Please choose from %s" %
                                   (eventcode, list(event_regex.keys())))
                required_event_regex[eventcode] = event_regex[eventcode]
            return required_event_regex
        return event_regex

    def get_columns(self) -> typing.Set[str]:
        """
        Get columns of windows event codes.

        Returns
        -------
        typing.Set[str]
            Columns of all configured eventcodes, if no interested eventcodes specified.
        """
        columns = set()
        for key in self.event_regex.keys():
            for column in self.event_regex[key].keys():
                columns.add(column)
        return columns
