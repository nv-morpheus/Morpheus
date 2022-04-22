#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e +o pipefail
# set -x
# set -v

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure to load the utils and globals
source ${SCRIPT_DIR}/val-globals.sh
source ${SCRIPT_DIR}/val-utils.sh

# Make sure the .tmp folder exists
if [[ ! -d "${MORPHEUS_ROOT}/.tmp" ]]; then
   mkdir -p "${MORPHEUS_ROOT}/.tmp"
fi

function run_pipeline_sid_minibert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --prefix="si_" \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize --exclude "id" --exclude "^_ts_" \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_sid_bert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-cased-hash.txt --truncation=True --do_lower_case=False --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --prefix="si_" \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_abp_nvsmi(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=${USE_CPP} \
      pipeline-fil \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_phishing_email(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=128 --labels_file=${MORPHEUS_ROOT}/data/labels_phishing.txt \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --label=pred --threshold=0.7 \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_hammah_user123(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=${USE_CPP} \
      pipeline-ae --userid_filter="user123" --userid_column_name="userIdentitysessionContextsessionIssueruserName" \
      from-cloudtrail --input_glob="${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-*.csv" \
      train-ae --train_data_glob="${MORPHEUS_ROOT}/models/datasets/training-data/hammah-*.csv" --seed 42 \
      preprocess \
      ${INFERENCE_STAGE} \
      add-scores \
      timeseries --resolution=1m --zscore_threshold=8.0 --hot_start \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --index_col="_index_" --exclude "event_dt" --rel_tol=0.1 --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_hammah_role-g(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=${USE_CPP} \
      pipeline-ae --userid_filter="role-g" --userid_column_name="userIdentitysessionContextsessionIssueruserName" \
      from-cloudtrail --input_glob="${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-*.csv" \
      train-ae --train_data_glob="${MORPHEUS_ROOT}/models/datasets/training-data/hammah-*.csv"  --seed 42 \
      preprocess \
      ${INFERENCE_STAGE} \
      add-scores \
      timeseries --resolution=10m --zscore_threshold=8.0 \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --index_col="_index_" --exclude "event_dt" --rel_tol=0.15 --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}
