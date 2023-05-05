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

MORPHEUS_MODULE_NAMESPACE = "morpheus"

DATA_LOADER = "DataLoader"
FILE_BATCHER = "FileBatcher"
FILE_TO_DF = "FileToDF"
FILTER_CONTROL_MESSAGE = "FilterControlMessage"
FILTER_DETECTIONS = "FilterDetections"
MLFLOW_MODEL_WRITER = "MLFlowModelWriter"
MULTIPLEXER = "Multiplexer"
SERIALIZE = "Serialize"
TO_CONTROL_MESSAGE = "ToControlMessage"
WRITE_TO_FILE = "WriteToFile"

SUPPORTED_DATA_TYPES = ["payload", "streaming"]
SUPPORTED_TASK_TYPES = ["load", "inference", "training"]
