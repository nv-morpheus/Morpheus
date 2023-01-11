#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-v] [-j] [-h] [-p] <model-name>

This script retrieves a computed model configuration from NVIDIA Triton. The Triton
Inference Server can be launched with "--strict-model-config=false" which means it
will create a minimal Triton configuration (config.pbtxt) from the required
elements of a model if one has not been provided.

This only applies to TensorRT, TensorFlow saved-model, and ONNX models. Once the
initial configuration file is derived, it is expected that optional and advanced
settings will be applied by hand.

By default, the script will take the JSON output from Triton and reformat it
to stdout for usage as a new configuration (i.e., save stdout to config.pbtxt).

Developed and tested with the 22.04 release of Triton using Morpheus ONNX and FIL
sample models only.

For more information on Triton model configuration, please refer to the online docs:
https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md
https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto

This script is intended for initial bootstrapping of a basic Triton model configuration.
Consult the Triton Model Navigator for advanced model optimization tooling.
https://github.com/triton-inference-server/model_navigator/blob/main/docs/optimize_for_triton.md

Available options:

-h, --help      Print this help and exit
-v, --verbose   Print debug info
-j, --json      Render the config directly in JSON without reformatting;
                useful for advanced jq manipulations
-t, --host 	Triton hostname or service endpoint name (default: localhost)
-p, --port      Triton REST API port (default: 8000)
EOF
  exit
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # script cleanup here
}

setup_colors() {
  if [[ -t 2 ]] && [[ -z "${NO_COLOR-}" ]] && [[ "${TERM-}" != "dumb" ]]; then
    NOFORMAT='\033[0m' RED='\033[0;31m' GREEN='\033[0;32m' ORANGE='\033[0;33m' BLUE='\033[0;34m' PURPLE='\033[0;35m' CYAN='\033[0;36m' YELLOW='\033[1;33m'
  else
    NOFORMAT='' RED='' GREEN='' ORANGE='' BLUE='' PURPLE='' CYAN='' YELLOW=''
  fi
}

msg() {
  echo >&2 -e "${1-}"
}

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}

parse_params() {
  json=0
  model=''
  host='localhost'
  port='8000'

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -v | --verbose) set -x ;;
    --no-color) NO_COLOR=1 ;;
    -j | --json) json=true ;;
    -t | --host) shift
                 host="$1"
                 ;;
    -p | --port) shift
                 port="$1"
                 ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  model=("$@")

  # check required params and arguments
  [[ ${#model[@]} -eq 0 ]] && die "Missing model argument"

  return 0
}

parse_params "$@"
setup_colors

which curl &>/dev/null || die "Please install curl"
which jq &>/dev/null || die "Please install jq"

# TODO: it may be possible to have one function to reformat
# both types of accelerators but it gets messy

# GPU accelerator parameters have a specific encoding
reformat_gpu_accelerator() {
        read jstream
        if echo $jstream | jq --exit-status -r '.optimization.execution_accelerators.gpu_execution_accelerator[]?.parameters? | keys[0]!=""' > /dev/null; then
	   echo "   gpu_execution_accelerator : [ {"
           NAME=$(echo $jstream | jq -r '.optimization.execution_accelerators.gpu_execution_accelerator[]?.name')
	   echo "    name: \"${NAME}\" "
           echo $jstream | jq -r \
                '.optimization.execution_accelerators.gpu_execution_accelerator[]?.parameters | to_entries[] | .key as $k | .value as $v | "\($k) \($v)"' | \
           awk '{ printf "    parameters { key: \"%s\" value: \"%s\" }\n", $1, $2 }'
	   echo "    } ]"
        fi
}

# should be just the name like 'openvino' and no parameters
reformat_cpu_accelerator() {
        read jstream
        if echo $jstream | jq --exit-status -r '.optimization.execution_accelerators.cpu_execution_accelerator[]?.name?!=""' > /dev/null; then
	   echo "   cpu_execution_accelerator : [ {"
           echo $jstream | jq -r \
                '.optimization.execution_accelerators.cpu_execution_accelerator[]?.name | values as $v | "\($v)"' | \
           awk '{ printf "   { name: \"%s\" },\n", $1 }'
	   echo "   } ]"
        fi
}

# optional parameters are freeform and encoded
# differently than accelerator parameters
reformat_optional() {
	read jstream
	if echo $jstream | jq --exit-status -r '.parameters? | keys[0]!=""' > /dev/null; then
	   echo $jstream | jq -r \
	      '.parameters? | to_entries[] | .key as $k | .value as $v | "\($k) \($v)"' | \
	   sed -E 's/\"(string_value)\"/\1/' | \
	   awk '{ printf "  { key: \"%s\" value: %s },\n", $1, $2 }'
	fi
}

chop_outer_braces() {
  sed -E '1d;$d'
}

left_indent() {
  sed -E 's/^  //'
}

unquote_keys() {
  sed -E 's/(^ *)"([^"]*)":/\1\2:/'
}

unquote_types() {
  sed -E 's/: "([A-Z0-9_]+)"/: \1/'
}

delete_empty() {
  jq -r 'del(..|select(. == ""))'
}

delete_optional() {
  jq -r 'del(.parameters?)'
}

delete_accelerators() {
  jq -r 'del(.optimization.execution_accelerators[]?)'
}

# try to retrieve the model config and die if model doesn't exist, etc.
RAW=$(mktemp)
HTTP_CODE=$(curl -sS --output ${RAW} --write-out "%{http_code}" ${host}:${port}/v2/models/${model}/config)
[[ ${HTTP_CODE} -lt 200 || ${HTTP_CODE} -gt 399 ]] && cat ${RAW} && rm ${RAW} && die "\nBad response"

if [ "$json" = true ]; then
   cat "${RAW}" | jq -r .
else
   PARAMS_GPU=$(cat ${RAW} | reformat_gpu_accelerator)
   PARAMS_CPU=$(cat ${RAW} | reformat_cpu_accelerator)
   PARAMS_OPT=$(cat ${RAW} | reformat_optional)
   # hack to tidy up last comma from jq transform
   PARAMS_OPT=$(echo "${PARAMS_OPT}" | sed '$s/,$//')
   STAGE_ZERO=$(cat ${RAW} | delete_empty)
   STAGE_ONE=$(echo ${STAGE_ZERO} | delete_optional)
   STAGE_TWO=$(echo ${STAGE_ONE} | delete_accelerators)

   # done with jq editing
   STAGE_THREE=$(echo ${STAGE_TWO} | jq -r . | chop_outer_braces | left_indent | unquote_keys | unquote_types)

   # after this point, we need to insert our pb text without jq
   # note the use of double-quotes for linefeeds
   if [[ ${PARAMS_GPU} || ${PARAMS_CPU} ]]; then
      STAGE_FOUR=$(echo "${STAGE_THREE}" | sed -E '/execution_accelerators/q' | sed -E '$d')
      STAGE_FIVE=$(echo "${STAGE_THREE}" | sed -n '/execution_accelerators/,$ p' | sed -E '1d')
      STAGE_LAST="${STAGE_FOUR}"$'\n  execution_accelerators {'

      # lay down each layer of the split and reformatted config
      if [[ ${PARAMS_GPU} ]]; then
         STAGE_LAST="${STAGE_LAST}"$'\n'"${PARAMS_GPU}"
      fi
      if [[ ${PARAMS_CPU} ]]; then
	 # insert comma if needed
	 [[ ${PARAMS_GPU} ]] && STAGE_LAST="${STAGE_LAST}"','
         STAGE_LAST="${STAGE_LAST}"$'\n'"${PARAMS_CPU}"
      fi
      # done with accelerators
      STAGE_LAST="${STAGE_LAST}"$'\n''  },'$'\n'"${STAGE_FIVE}"
   else
      # no accelerators, pivot back to stage 3
      STAGE_LAST="${STAGE_THREE}"
   fi
   if [[ ${PARAMS_OPT} ]]; then
      STAGE_LAST="${STAGE_LAST}"','$'\n''parameters ['"${PARAMS_OPT}"' ]'
   fi
   echo "${STAGE_LAST}"
fi

# cleanup
rm ${RAW}
