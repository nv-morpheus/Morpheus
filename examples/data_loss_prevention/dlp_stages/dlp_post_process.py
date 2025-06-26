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

from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.stage_decorator import stage


@stage(name="dlp-post-process", execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
def dlp_post_process(msg: ControlMessage,
                     *,
                     output_columns: list[str],
                     sort_col: str = 'original_source_index') -> MessageMeta:
    """
    Convert the incoming ControlMessage payload to a sorted MessageMeta containing only the requested columns.
    """
    with msg.payload().mutable_dataframe() as df:
        new_meta = MessageMeta(df[output_columns].sort_values(sort_col))

    return new_meta
