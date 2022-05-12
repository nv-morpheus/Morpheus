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

source ci/scripts/jenkins_common.sh

gpuci_logger "Creating conda env"
conda config --add pkgs_dirs /opt/conda/pkgs
conda config --env --add channels conda-forge
conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS:-"https://conda.anaconda.org"}
mamba install -q -y -n base -c conda-forge "boa >=0.10"
mamba create -q -y -n morpheus python=${PYTHON_VER}
conda activate morpheus

gpuci_logger "Installing CI dependencies"
mamba env update -q -n morpheus -f ./docker/conda/environments/cuda${CUDA_VER}_ci.yml

gpuci_logger "Check versions"
python3 --version
gcc --version
g++ --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Checking S3 cuDF cache"
CONDA_BLD_DIR=/opt/conda/conda-bld
CUDF_CONDA_COMMIT=$(git log -n 1 --pretty=format:%H -- ci/conda)
CUDF_CONDA_CACHE_PATH="/cudf/${CUDA_VER}/${PYTHON_VER}/${RAPIDS_VER}/${CUDF_CONDA_COMMIT}/${NVARCH}/cudf_conda.tar.gz"
CUDF_CONDA_CACHE_URL="${S3_URL}${CUDF_CONDA_CACHE_PATH}"
CUDF_CONDA_TAR="${WORKSPACE_TMP}/cudf_conda.tar.bz"

echo "Checking ${DISPLAY_URL}${CUDF_CONDA_CACHE_PATH}"
set +e
aws s3 cp --no-progress ${CUDF_CONDA_CACHE_URL} ${CUDF_CONDA_TAR}
CUDF_CACHE_CHECK=$?
set -e

if [[ "${CUDF_CACHE_CHECK}" != "0" ]]; then
      gpuci_logger "Cache miss, Building cuDF"
      mkdir -p ${CONDA_BLD_DIR}
      # The --no-build-id bit is needed for sccache
      USE_SCCACHE=1 CONDA_ARGS="--no-build-id --output-folder ${CONDA_BLD_DIR} --skip-existing --no-test" ${MORPHEUS_ROOT}/ci/conda/recipes/run_conda_build.sh libcudf cudf

      gpuci_logger "sccache usage for cudf build:"
      sccache --show-stats
      ZS=$(sccache --zero-stats)

      gpuci_logger "Archiving cuDF build"
      cd $(dirname ${CONDA_BLD_DIR})
      tar cfj ${CUDF_CONDA_TAR} $(basename ${CONDA_BLD_DIR})
      cd -
      aws s3 cp --no-progress ${CUDF_CONDA_TAR} ${CUDF_CONDA_CACHE_URL}
else
      gpuci_logger "Cache hit, using cached cuDF"
      cd $(dirname ${CONDA_BLD_DIR})
      tar xf ${CUDF_CONDA_TAR}
      cd -
fi

gpuci_logger "Installing cuDF"
mamba install -q -y -c file://${CONDA_BLD_DIR} -c nvidia -c rapidsai -c conda-forge libcudf cudf

gpuci_logger "Installing other dependencies"
mamba env update -q -n morpheus -f ./docker/conda/environments/cuda${CUDA_VER}_dev.yml

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
      -DMORPHEUS_PYTHON_INPLACE_BUILD=ON \
      -DMORPHEUS_USE_CCACHE=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache \
      .

gpuci_logger "Building Morpheus"
cmake --build build -j --parallel ${PARALLEL_LEVEL}

gpuci_logger "sccache usage for morpheus build:"
sccache --show-stats

gpuci_logger "Installing Morpheus"
pip install -e ${MORPHEUS_ROOT}

gpuci_logger "Archiving results"
mamba pack --quiet --force --ignore-editable-packages --ignore-missing-files --n-threads ${PARALLEL_LEVEL} -n morpheus -o ${WORKSPACE_TMP}/conda_env.tar.gz
tar cfj ${WORKSPACE_TMP}/workspace.tar.bz --exclude=".git" --exclude="models" --exclude=".cache" ./
ls -lh ${WORKSPACE_TMP}/

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/conda_env.tar.gz" "${ARTIFACT_URL}/conda_env.tar.gz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/workspace.tar.bz" "${ARTIFACT_URL}/workspace.tar.bz"

# gpuci_logger "Running conda build for morpheus"
# ZS=$(sccache --zero-stats)
# USE_SCCACHE=1 CONDA_ARGS="--no-build-id --output-folder ${CONDA_BLD_DIR} --no-test" ${MORPHEUS_ROOT}/ci/conda/recipes/run_conda_build.sh morpheus

# gpuci_logger "sccache usage for morpheus conda build:"
# sccache --show-stats

# gpuci_logger "Archiving conda builds"
# tar cfj ${WORKSPACE_TMP}/conda_build.tar.bz ${CONDA_BLD_DIR}
# ls -lh ${WORKSPACE_TMP}/

# gpuci_logger "Pushing conda builds to ${DISPLAY_ARTIFACT_URL}"
# aws s3 cp --no-progress "${WORKSPACE_TMP}/conda_build.tar.bz" "${ARTIFACT_URL}/conda_build.tar.bz"

gpuci_logger "Success"
exit 0
