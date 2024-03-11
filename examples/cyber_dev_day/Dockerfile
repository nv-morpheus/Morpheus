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

ARG MORPHEUS_CONTAINER=nvcr.io/nvidia/morpheus/morpheus
ARG MORPHEUS_CONTAINER_VERSION=v24.03.00-dev

FROM ${MORPHEUS_CONTAINER}:${MORPHEUS_CONTAINER_VERSION} as base

WORKDIR /workspace

# # Use a separate build directory to avoid clashes on the host
# ENV BUILD_DIR=build-docker

# Make it easier to run commands in the container
ENV SHELL=/bin/bash

# Copy over just the environment file to install the dependencies without busting the cache from source file changes
COPY ./examples/cyber_dev_day/requirements.yaml ./examples/cyber_dev_day/

# Install the example's dependencies
RUN source activate morpheus \
    && mamba env update -n morpheus -f ./examples/cyber_dev_day/requirements.yaml

# Copy everything over to the container to build
COPY . ./

# Build morpheus
RUN source activate morpheus &&\
    CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_PYTHON_BUILD_STUBS=OFF -DMORPHEUS_PYTHON_PERFORM_INSTALL=ON -DMORPHEUS_CUDA_ARCHITECTURES=75;80" ./scripts/compile.sh

# Set the workdir to the dev day example
WORKDIR /workspace/examples/cyber_dev_day/

# # If any changes have been made from the base image, recopy the sources
# COPY ./examples/cyber_dev_day /workspace/examples/cyber_dev_day/

# ===== Setup for running unattended =====
FROM base as runtime

# # Install the Workflow package
# RUN source activate morpheus \
#     && pip install -e .

# # Keep the container running indefinitely
# CMD ["sleep", "infinity"]

# ===== Setup for running Jupyter =====
FROM base as jupyter

# Install the jupyter specific requirements
RUN source activate morpheus &&\
    mamba install -y -c conda-forge \
        ipywidgets \
        jupyter_contrib_nbextensions \
        # notebook v7 is incompatible with jupyter_contrib_nbextensions
        notebook=6 &&\
    jupyter contrib nbextension install --user &&\
    pip install jupyterlab_nvdashboard==0.9

# Launch jupyter
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
