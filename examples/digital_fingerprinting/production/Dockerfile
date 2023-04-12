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
ARG MORPHEUS_CONTAINER_VERSION=v23.07.00-runtime

FROM ${MORPHEUS_CONTAINER}:${MORPHEUS_CONTAINER_VERSION} as base

# # Fix the entrypoint to work with different WORKDIR
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

SHELL ["/bin/bash", "-c"]

# Set the workdir to the DFP base folder
WORKDIR /workspace/examples/digital_fingerprinting/

# Install Node.js for DFP Viz
RUN apt-get update \
    && apt-get upgrade -y \
    && curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install --no-install-recommends -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Add library path where DFP Viz can find libnvrtc
ENV LD_LIBRARY_PATH=/opt/conda/envs/morpheus/lib:$LD_LIBRARY_PATH

# Copy the conda_env file
COPY ./conda_env.yml ./

# Install DFP dependencies
RUN source activate morpheus \
    && mamba env update -n morpheus -f ./conda_env.yml

# Set the tracking URI for mlflow
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

# This will get used by pipelines for the --s3_cache option
# ENV DFP_S3_CACHE="/work/examples/dfp_workflow/morpheus/.s3_cache"

# If any changes have been made from the base image, recopy the sources
COPY . /workspace/examples/digital_fingerprinting/

# Set the workdir to be the morpheus folder
WORKDIR /workspace/examples/digital_fingerprinting/production/morpheus

# ===== Setup for running unattended =====
FROM base as runtime

# Launch morpheus
CMD ["./launch.sh"]

# ===== Setup for running Jupyter =====
FROM base as jupyter

# Install the jupyter specific requirements
RUN source activate morpheus \
    && mamba install -y -c conda-forge \
        ipywidgets \
        nb_conda_kernels \
    && pip install jupyter_contrib_nbextensions==0.5.1 \
    && jupyter contrib nbextension install --user \
    && pip install jupyterlab_nvdashboard==0.7.0

# Launch jupyter
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
