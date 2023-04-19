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
from abc import ABC
from abc import abstractmethod

import yaml

import cudf

log = logging.getLogger(__name__)


class EventParser(ABC):
    """This is an abstract class for all event log parsers.

    :param columns: Event column names.
    :type columns: set(string)
    :param event_name: Event name
    :type event_name: string
    """

    def __init__(self, columns, event_name):
        """Constructor method
        """
        self._columns = columns
        self._event_name = event_name

    @property
    def columns(self):
        """List of columns that are being processed.

        :return: Event column names.
        :rtype: set(string)
        """
        return self._columns

    @property
    def event_name(self):
        """Event name define type of logs that are being processed.

        :return: Event name
        :rtype: string
        """
        return self._event_name

    @abstractmethod
    def parse(self, text):
        """
        Abstract method 'parse' triggers the parsing functionality. Subclasses are required to implement
        and execute any parsing pre-processing steps.
        """
        log.info("Begin parsing of dataframe")
        pass

    def parse_raw_event(self, text, event_regex):
        """Processes parsing of a specific type of raw event records received as a dataframe.

        :param dataframe: Raw events to be parsed.
        :type dataframe: cudf.DataFrame
        :param raw_column: Raw data contained column name.
        :type raw_column: string
        :param event_regex: Required regular expressions for a given event type.
        :type event_regex: dict
        :return: parsed information.
        :rtype: cudf.DataFrame
        """
        log.debug("Parsing raw events. Event type: " + self.event_name)

        parsed_gdf = cudf.DataFrame({col: [""] for col in self.columns})
        parsed_gdf = parsed_gdf[:0]
        event_specific_columns = event_regex.keys()
        # Applies regex pattern for each expected output column to raw data
        for col in event_specific_columns:
            regex_pattern = event_regex.get(col)
            extracted_gdf = text.str.extract(regex_pattern)
            if not extracted_gdf.empty:
                parsed_gdf[col] = extracted_gdf[0]

        remaining_columns = list(self.columns - event_specific_columns)
        # Fill remaining columns with empty.
        for col in remaining_columns:
            parsed_gdf[col] = ""

        return parsed_gdf

    def _load_regex_yaml(self, yaml_file):
        """Returns a dictionary of the regex contained in the given yaml file"""
        with open(yaml_file) as yaml_file:
            regex_dict = yaml.safe_load(yaml_file)
        return regex_dict
