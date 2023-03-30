# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import mrc
import pandas as pd
from dfp.utils.logging_timer import log_time
from mrc.core import operators as ops

import cudf

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_SPLIT_USERS

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_SPLIT_USERS, MORPHEUS_MODULE_NAMESPACE)
def dfp_split_users(builder: mrc.Builder):
    """
    This module function splits data based on user IDs.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline builder instance.

    Notes
    -----
    Configurable parameters:
        - fallback_username (str): The user ID to use if the user ID is not found; Example: "generic_user";
        Default: 'generic_user'
        - include_generic (bool): Whether to include a generic user ID in the output; Example: false; Default: False
        - include_individual (bool): Whether to include individual user IDs in the output; Example: true; Default: False
        - only_users (list): List of user IDs to include; others will be excluded; Example: ["user1", "user2", "user3"];
         Default: []
        - skip_users (list): List of user IDs to exclude from the output; Example: ["user4", "user5"]; Default: []
        - timestamp_column_name (str): Name of the column containing timestamps; Example: "timestamp";
        Default: 'timestamp'
        - userid_column_name (str): Name of the column containing user IDs; Example: "username"; Default: 'username'
    """

    config = builder.get_current_module_config()

    skip_users = config.get("skip_users", [])
    only_users = config.get("only_users", [])

    timestamp_column_name = config.get("timestamp_column_name", "timestamp")
    userid_column_name = config.get("userid_column_name", "username")
    include_generic = config.get("include_generic", False)
    include_individual = config.get("include_individual", False)

    if (include_generic):
        # if not "fallback_username" in config:
        #    raise ValueError("fallback_username must be specified if include_generic is True")
        fallback_username = config.get("fallback_username", "generic_user")

    # Map of user ids to total number of messages. Keep indexes monotonic and increasing per user
    user_index_map: typing.Dict[str, int] = {}

    def generate_control_messages(control_message: ControlMessage, split_dataframes: typing.Dict[str, cudf.DataFrame]):
        output_messages: typing.List[ControlMessage] = []

        for user_id in sorted(split_dataframes.keys()):
            if (user_id in skip_users):
                continue

            user_df = split_dataframes[user_id]

            current_user_count = user_index_map.get(user_id, 0)

            # Reset the index so that users see monotonically increasing indexes
            user_df.index = range(current_user_count, current_user_count + len(user_df))
            user_index_map[user_id] = current_user_count + len(user_df)

            user_control_message = control_message.copy()
            user_control_message.set_metadata("user_id", user_id)

            user_cudf = cudf.from_pandas(user_df)
            user_control_message.payload(MessageMeta(df=user_cudf))

            output_messages.append(user_control_message)

        return output_messages

    def generate_split_dataframes(df: pd.DataFrame):
        split_dataframes: typing.Dict[str, cudf.DataFrame] = {}

        # If we are skipping users, do that here
        if (len(skip_users) > 0):
            df = df[~df[userid_column_name].isin(skip_users)]

        if (len(only_users) > 0):
            df = df[df[userid_column_name].isin(only_users)]

        # Split up the dataframes
        if (include_generic):
            split_dataframes[fallback_username] = df

        if (include_individual):
            split_dataframes.update(
                {username: user_df
                 for username, user_df in df.groupby(userid_column_name, sort=False)})

        return split_dataframes

    def extract_users(control_message: ControlMessage):
        # logger.debug("Extracting users from message")
        if (control_message is None):
            logger.debug("No message to extract users from")
            return []

        try:
            control_messages = None  # for readability
            mm = control_message.payload()
            with mm.mutable_dataframe() as dfm:
                with log_time(logger.debug):

                    if (isinstance(dfm, cudf.DataFrame)):
                        # Convert to pandas because cudf is slow at this
                        df = dfm.to_pandas()
                        df[timestamp_column_name] = pd.to_datetime(df[timestamp_column_name], utc=True)
                    else:
                        df = dfm

                    split_dataframes = generate_split_dataframes(df)

                    control_messages = generate_control_messages(control_message, split_dataframes)

            return control_messages
        except Exception:
            logger.exception("Error extracting users from message, discarding control message")
            return []

    def node_fn(observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.pipe(ops.map(extract_users), ops.flatten()).subscribe(subscriber)

    node = builder.make_node(DFP_SPLIT_USERS, mrc.core.operators.build(node_fn))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
