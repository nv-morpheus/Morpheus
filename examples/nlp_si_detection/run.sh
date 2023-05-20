# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

morpheus --log_level=DEBUG run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=32 pipeline-nlp --model_seq_length=256 from-file --filename=/workspace/examples/data/pcap_dump.jsonlines deserialize preprocess --vocab_hash_file=/workspace/models/training-tuning-scripts/sid-models/resources/bert-base-uncased-hash.txt --do_lower_case=True --truncation=True inf-triton --model_name=sid-minibert-onnx --server_url=localhost:8000 --force_convert_inputs=True monitor --description "Inference Rate" --smoothing=0.001 --unit inf add-class serialize --exclude '^_ts_' to-file --filename=detections.jsonlines --overwrite
