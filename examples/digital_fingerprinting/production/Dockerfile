# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMG=nvcr.io/nvidia/cuda
ARG BASE_IMG_TAG=12.5.1-base-ubuntu22.04

FROM ${BASE_IMG}:${BASE_IMG_TAG} as base

# Install necessary dependencies using apt-get
RUN apt-get update && apt-get install -y \
      wget \
    && apt-get clean

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Install Mamba, a faster alternative to conda, within the base environment
RUN conda install -y mamba -n base -c conda-forge

# Create base environment
RUN conda create -y --name morpheus-dfp python=3.10

# Activate the environment (make it default for subsequent commands)
RUN echo "source activate morpheus-dfp" >> ~/.bashrc

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Add conda channels required for the Morpheus dependencies
RUN source activate morpheus-dfp \
    && conda config --env --add channels conda-forge \
    && conda config --env --add channels nvidia \
    && conda config --env --add channels rapidsai \
    && conda config --env --add channels pytorch

# Install morpheus-llm conda package
RUN source activate morpheus-dfp \
    && mamba install -y \
     -c nvidia/label/dev morpheus-dfp

RUN source activate morpheus-dfp \
     && mamba install -y -c conda-forge tini=0.19

# # Fix the entrypoint to work with different WORKDIR
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

SHELL ["/bin/bash", "-c"]

# Set the workdir to the DFP base folder
WORKDIR /workspace/examples/digital_fingerprinting/

COPY ./example_dfp_cuda-125_arch-x86_64.yaml /workspace/examples/digital_fingerprinting/production/

# Install DFP dependencies
RUN source activate morpheus-dfp \
    && /opt/conda/bin/conda env update --solver=libmamba -n morpheus-dfp --file /workspace/examples/digital_fingerprinting/production/example_dfp_cuda-125_arch-x86_64.yaml

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
RUN source activate morpheus-dfp &&\
    /opt/conda/bin/conda install --solver=libmamba -y -c conda-forge \
        ipywidgets \
        jupyter_contrib_nbextensions \
        # notebook v7 is incompatible with jupyter_contrib_nbextensions
        notebook=6 &&\
    jupyter contrib nbextension install --user &&\
    pip install jupyterlab_nvdashboard==0.9

# Launch jupyter
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
