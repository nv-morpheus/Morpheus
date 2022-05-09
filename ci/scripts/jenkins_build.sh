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
conda install -q -y -n base -c conda-forge "boa >=0.10" python=${PYTHON_VER}
conda create -q -y -n morpheus python=${PYTHON_VER}
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

CONDA_BLD_DIR=/opt/conda/conda-bld

gpuci_logger "Checking S3 cuDF cache"
CUDF_CONDA_COMMIT=$(git log -n 1 --pretty=format:%H -- ci/conda)
CUDF_CONDA_CACHE_URL="${S3_URL}/${CUDA_VER}/${PYTHON_VER}/${RAPIDS_VER}/${CUDF_CONDA_COMMIT}/cudf_conda.tar.gz"
CUDF_CONDA_TAR="${WORKSPACE_TMP}/cudf_conda.tar.gz"

set +e
aws s3 ls ${CUDF_CONDA_CACHE_URL}
CACHE_CHECK=$?
set -e

if [[ "${CACHE_CHECK}" != "0" ]]; then
      gpuci_logger "Cache miss, Building cuDF"
      # The --no-build-id bit is needed for sccache
      USE_SCCACHE=1 CONDA_ARGS="--no-build-id --output-folder ${CONDA_BLD_DIR} --skip-existing --no-test" ${MORPHEUS_ROOT}/ci/conda/recipes/run_conda_build.sh libcudf cudf

      gpuci_logger "sccache usage for cudf build:"
      sccache --show-stats
      ZS=$(sccache --zero-stats)

      gpuci_logger "Archiving cuDF build"
      tar cfz ${CUDF_CONDA_TAR} ${CONDA_BLD_DIR}
      aws_cp ${CUDF_CONDA_TAR} ${CUDF_CONDA_CACHE_URL}
else
      gpuci_logger "Cache hit, using cached cuDF"
      aws_cp ${CUDF_CONDA_CACHE_URL} ${CUDF_CONDA_TAR}
      tar xf ${CUDF_CONDA_TAR} --directory /opt/conda
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
cmake --build build -j

gpuci_logger "sccache usage for morpheus build:"
sccache --show-stats

gpuci_logger "Installing Morpheus"
pip install -e ${MORPHEUS_ROOT}

gpuci_logger "Archiving results"
mamba pack --quiet --force --ignore-missing-files --ignore-editable-packages --n-threads ${PARALLEL_LEVEL} -n morpheus -o ${WORKSPACE_TMP}/conda.tar.gz
tar cfz ${WORKSPACE_TMP}/build.tar.gz build

gpuci_logger "Pushing results to S3"
aws_cp ${WORKSPACE_TMP}/conda.tar.gz "${ARTIFACT_URL}/conda.tar.gz"
aws_cp ${WORKSPACE_TMP}/build.tar.gz "${ARTIFACT_URL}/build.tar.gz"
