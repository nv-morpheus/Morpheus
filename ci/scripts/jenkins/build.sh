#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

source ${WORKSPACE}/ci/scripts/jenkins/common.sh

gpuci_logger "Creating conda env"
rm -rf ${MORPHEUS_ROOT}/.cache/ ${MORPHEUS_ROOT}/build/
conda config --add pkgs_dirs /opt/conda/pkgs
conda config --env --add channels conda-forge
conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS:-"https://conda.anaconda.org"}
mamba install -q -y -n base -c conda-forge "boa >=0.10"
mamba create -q -y -n morpheus python=${PYTHON_VER}
conda activate morpheus

gpuci_logger "Installing CI dependencies"
mamba env update -q -n morpheus -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_ci.yml

gpuci_logger "Check versions"
python3 --version
gcc --version
g++ --version

show_conda_info

gpuci_logger "Checking S3 cuDF cache"
CUDF_CONDA_BLD_DIR=/opt/conda/conda-bld
CUDF_CONDA_COMMIT=$(git log -n 1 --pretty=format:%H -- ci/conda)
CUDF_CONDA_CACHE_PATH="/cudf/${CUDA_VER}/${PYTHON_VER}/${RAPIDS_VER}/${CUDF_CONDA_COMMIT}/${NVARCH}/cudf_conda.tar.bz"
CUDF_CONDA_CACHE_URL="${S3_URL}${CUDF_CONDA_CACHE_PATH}"
CUDF_CONDA_TAR="${WORKSPACE_TMP}/cudf_conda.tar.bz"

gpuci_logger "Checking ${DISPLAY_URL}${CUDF_CONDA_CACHE_PATH}"
set +e
fetch_s3 "${CUDF_CONDA_CACHE_PATH}" "${CUDF_CONDA_TAR}"
set -e

if [[ "${FETCH_STATUS}" != "0" ]]; then
      gpuci_logger "Cache miss, Building cuDF"
      mkdir -p ${CUDF_CONDA_BLD_DIR}
      # The --no-build-id bit is needed for sccache
      CONDA_ARGS="--no-build-id --output-folder ${CUDF_CONDA_BLD_DIR} --skip-existing --no-test" ${MORPHEUS_ROOT}/ci/conda/recipes/run_conda_build.sh libcudf cudf

      gpuci_logger "sccache usage for cudf build:"
      sccache --show-stats
      sccache --zero-stats 2>&1 > /dev/null

      gpuci_logger "Archiving cuDF build"
      cd $(dirname ${CUDF_CONDA_BLD_DIR})
      tar cfj ${CUDF_CONDA_TAR} $(basename ${CUDF_CONDA_BLD_DIR})
      cd -
      aws s3 cp --no-progress ${CUDF_CONDA_TAR} ${CUDF_CONDA_CACHE_URL}
else
      gpuci_logger "Cache hit, using cached cuDF"
      cd $(dirname ${CUDF_CONDA_BLD_DIR})
      tar xf ${CUDF_CONDA_TAR}
      cd -
fi

gpuci_logger "Installing cuDF"
mamba install -q -y -c local -c nvidia -c rapidsai -c conda-forge libcudf cudf

gpuci_logger "Installing other dependencies"
mamba env update -q -n morpheus -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_dev.yml
conda deactivate && conda activate morpheus

gpuci_logger "Check cmake & ninja"
cmake --version
ninja --version

gpuci_logger "Configuring cmake for Morpheus"
cmake -B build -G Ninja \
      -DCMAKE_MESSAGE_CONTEXT_SHOW=ON \
      -DMORPHEUS_BUILD_BENCHMARKS=ON \
      -DMORPHEUS_BUILD_EXAMPLES=ON \
      -DMORPHEUS_BUILD_TESTS=ON \
      -DMORPHEUS_USE_CONDA=ON \
      -DMORPHEUS_PYTHON_INPLACE_BUILD=OFF \
      -DMORPHEUS_USE_CCACHE=ON \
      -DCCACHE_PROGRAM_PATH=$(which sccache) \
      .

gpuci_logger "Building Morpheus"
cmake --build build -j --parallel ${PARALLEL_LEVEL}

gpuci_logger "sccache usage for morpheus build:"
sccache --show-stats

gpuci_logger "Installing Morpheus"
cmake -DCOMPONENT=Wheel -P ${MORPHEUS_ROOT}/build/cmake_install.cmake
pip install ${MORPHEUS_ROOT}/build/wheel

gpuci_logger "Archiving results"
mamba pack --quiet --force --ignore-missing-files --n-threads ${PARALLEL_LEVEL} -n morpheus -o ${WORKSPACE_TMP}/conda_env.tar.gz

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/conda_env.tar.gz" "${ARTIFACT_URL}/conda_env.tar.gz"

gpuci_logger "Success"
exit 0
