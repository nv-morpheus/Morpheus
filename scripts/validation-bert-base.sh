#!/bin/bash

set -e +o pipefail
set -x
set -v

# Color variables
b="\033[0;36m"
g="\033[0;32m"
r="\033[0;31m"
e="\033[0;90m"
y="\033[0;33m"
x="\033[0m"

# RUN OPTIONS
RUN_DOCKER=${RUN_DOCKER:-1}
RUN_PYTORCH=${RUN_PYTORCH:-1}
RUN_ONNX=${RUN_ONNX:-1}
RUN_TENSORRT=${RUN_TENSORRT:-1}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MORPHEUS_ROOT=$(realpath ${MORPHEUS_ROOT:-"${SCRIPT_DIR}/.."})

PCAP_INPUT_FILE=${PCAP_INPUT_FILE:-"${MORPHEUS_ROOT}/data/pcap_training_data.jsonlines"}
PCAP_TRUTH_FILE=${PCAP_TRUTH_FILE:-"${MORPHEUS_ROOT}/data/pcap_training_data-truth.jsonlines"}

MODEL_PYTORCH_FILE=${MODEL_PYTORCH_FILE:-"${MORPHEUS_ROOT}/models/sid-models/sid-bert-20210610.pth"}
MODEL_DIRECTORY=${MODEL_PYTORCH_FILE%/*}
MODEL_FILENAME=$(basename -- "${MODEL_PYTORCH_FILE}")
MODEL_EXTENSION="${MODEL_FILENAME##*.}"
MODEL_NAME="${MODEL_FILENAME%.*}"

OUTPUT_PYTORCH_FILE="${MORPHEUS_ROOT}/.tmp/val_${MODEL_NAME}-pytorch.jsonlines"
OUTPUT_ONNX_FILE="${MORPHEUS_ROOT}/.tmp/val_${MODEL_NAME}-onnx.jsonlines"
OUTPUT_TRT_FILE="${MORPHEUS_ROOT}/.tmp/val_${MODEL_NAME}-trt.jsonlines"

if [[ "${RUN_DOCKER}" = "1" ]]; then
   # Kill the triton container on exit
   function cleanup {
   echo "Killing Triton container"
   docker kill triton-validation
   }

   trap cleanup EXIT
fi

function wait_for_triton {

   max_retries=${GPUCI_RETRY_MAX:=3}
   retries=0
   sleep_interval=${GPUCI_RETRY_SLEEP:=1}

   res=$(curl -X GET \
        --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 2 \
        "http://localhost:8000/v2/health/ready" || echo 'HTTP 500')

   status=$(echo "$res" | head -1 | cut -d' ' -f2)

   while [[ "$status" != "" ]] && [[ "$status" != "200" ]] && \
         (( ${retries} < ${max_retries} )); do
      echo -e "${y}Triton not ready. Waiting ${sleep_interval} sec...${x}"

      sleep ${sleep_interval}

      res=$(curl -X GET \
         --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 2 \
         "http://localhost:8000/v2/health/ready" || echo 'HTTP 500')

      status=$(echo "$res" | head -1 | cut -d' ' -f2)
      retries=$((retries+1))
   done

   echo -e "${g}Triton is ready.${x}"
   return 0
}

if [[ "${RUN_PYTORCH}" = "1" ]]; then
   # Run the script for PyTorch
   morpheus --log_level=DEBUG --debug run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=512 pipeline --model_seq_length=256 \
      from-file --filename=${PCAP_INPUT_FILE} \
      buffer \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-cased-hash.txt --truncation=True --do_lower_case=False --add_special_tokens=True \
      buffer \
      inf-pytorch --model_filename=${MODEL_PYTORCH_FILE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      serialize \
      to-file --filename=${OUTPUT_PYTORCH_FILE} --overwrite

   # Print the diff
   echo "PyTorch Error Count: $(diff ${PCAP_TRUTH_FILE} ${OUTPUT_PYTORCH_FILE} | grep "^>" | wc -l)"
else
   echo -e "${y}Skipping PyTorch${x}"
fi


if [[ "${RUN_DOCKER}" = "1" ]]; then
   echo "Launching Triton Container"

   TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:21.02-py3"}

   # Launch triton container in explicit mode
   docker run --rm -ti -d --name=triton-validation --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v ${MORPHEUS_ROOT}/models:/models ${TRITON_IMAGE} tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit
fi

wait_for_triton

if [[ "${RUN_PYTORCH}" = "1" ]]; then
   # Tell Triton to load the ONNX model
   curl --request POST \
   --url http://localhost:8000/v2/repository/models/sid-bert-onnx/load

   echo "ONNX model loaded in Triton"

   # Run the script for ONNX
   morpheus --log_level=DEBUG --debug run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=32 pipeline --model_seq_length=256 \
      from-file --filename=${PCAP_INPUT_FILE} \
      buffer \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-cased-hash.txt --truncation=True --do_lower_case=False --add_special_tokens=True \
      buffer \
      inf-triton --model_name=sid-bert-onnx --server_url=localhost:8001 --force_convert_inputs=True \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      serialize \
      to-file --filename=${OUTPUT_ONNX_FILE} --overwrite

   # Print the diff
   echo "ONNX Error Count: $(diff ${PCAP_TRUTH_FILE} ${OUTPUT_ONNX_FILE} | grep "^>" | wc -l)"
else
   echo -e "${y}Skipping ONNX${x}"
fi

if [[ "${RUN_TENSORRT}" = "1" ]]; then
   # Generate the TensorRT model
   cd ${MORPHEUS_ROOT}/models/triton-model-repo/sid-bert-trt/1

   echo "Generating the TensorRT model. This may take a minute..."
   morpheus tools onnx-to-trt --input_model ${MODEL_DIRECTORY}/${MODEL_NAME}.onnx --output_model ./sid-bert-trt_b1-8_b1-16_b1-32.engine --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 256 --max_workspace_size 16000

   cd ${MORPHEUS_ROOT}

   # Tell Triton to load the model
   curl --request POST \
   --url http://localhost:8000/v2/repository/models/sid-bert-trt/load

   # Run the script for TensorRT
   morpheus --log_level=DEBUG --debug run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=32 pipeline --model_seq_length=256 \
      from-file --filename=${PCAP_INPUT_FILE} \
      buffer \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-cased-hash.txt --truncation=True --do_lower_case=False --add_special_tokens=True \
      buffer \
      inf-triton --model_name=sid-bert-trt --server_url=localhost:8001 --force_convert_inputs=True \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      serialize \
      to-file --filename=${OUTPUT_TRT_FILE} --overwrite

   echo "TensorRT Error Count: $(diff ${PCAP_TRUTH_FILE} ${OUTPUT_TRT_FILE} | grep "^>" | wc -l)"
else
   echo -e "${y}Skipping TensorRT${x}"
fi

echo -e "${g}Complete!${x}"
