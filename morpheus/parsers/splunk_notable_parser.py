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

import cudf

from morpheus.parsers.event_parser import EventParser

log = logging.getLogger(__name__)


class SplunkNotableParser(EventParser):
    """This is class parses splunk notable logs.
    """
    REGEX_FILE = "resources/splunk_notable_regex.yaml"
    EVENT_NAME = "notable"

    def __init__(self):
        """Constructor method
        """
        event_regex = {}
        regex_filepath = (os.path.dirname(os.path.abspath(__file__)) + "/" + self.REGEX_FILE)
        self.event_regex = self._load_regex_yaml(regex_filepath)
        EventParser.__init__(self, event_regex.keys(), self.EVENT_NAME)

    def parse(self, dataframe, raw_column):
        """Parses the Splunk notable raw events.

        :param dataframe: Raw events to be parsed.
        :type dataframe: cudf.DataFrame
        :param raw_column: Raw data contained column name.
        :type raw_column: string
        :return: parsed information.
        :rtype: cudf.DataFrame
        """
        # Cleaning raw data to be consistent.
        dataframe[raw_column] = dataframe[raw_column].str.replace("\\\\", "")
        parsed_dataframe = self.parse_raw_event(dataframe, raw_column, self.event_regex)
        # Replace null values of all columns with empty.
        parsed_dataframe = parsed_dataframe.fillna("")
        # Post-processing: for src_ip and dest_ip.
        parsed_dataframe = self._process_ip_fields(parsed_dataframe)
        return parsed_dataframe

    def _process_ip_fields(self, parsed_dataframe):
        """
        This function replaces src_ip column with src_ip2, if scr_ip is empty and does the same way for dest_ip column.
        """
        for ip in ["src_ip", "dest_ip"]:
            log.debug("******* Processing %s *******" % (ip))
            ip2 = ip + "2"
            ip_len = ip + "_len"
            # Calculate ip column value length.
            parsed_dataframe[ip_len] = parsed_dataframe[ip].str.len()
            # Retrieve empty ip column records.
            tmp_dataframe = parsed_dataframe[parsed_dataframe[ip_len] == 0]
            # Retrieve non empty ip column records.
            parsed_dataframe = parsed_dataframe[parsed_dataframe[ip_len] != 0]

            if not tmp_dataframe.empty:
                log.debug("tmp_dataframe size %s" % (str(tmp_dataframe.shape)))
                # Assign ip2 column values to empty ip column values.
                tmp_dataframe[ip] = tmp_dataframe[ip2]
                if not parsed_dataframe.empty:
                    log.debug("parsed_dataframe is not empty %s" % (str(parsed_dataframe.shape)))
                    # Concat, if both parsed_dataframe and tmp_df are not empty.
                    parsed_dataframe = cudf.concat([parsed_dataframe, tmp_dataframe])
                else:
                    # If parsed_dataframe is empty assign tmp_df.
                    parsed_dataframe = tmp_dataframe
            # Remove ip2 and ip_len columns. Since data is captured in ip column.
            parsed_dataframe = parsed_dataframe.drop([ip_len, ip2], axis=1)
        return parsed_dataframe
