FROM nvidia/cuda:11.5.1-devel-ubuntu20.04 AS base

# FROM ghcr.io/trxcllnt/devcontainer-images/cmake-ninja-sccache-llvm-cuda-mambaforge:latest AS base

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
        wget git-lfs gdb iputils-ping

ENV CONDA_DIR=/opt/conda

RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR} && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

ENV PATH="${CONDA_DIR}/bin:${PATH}"

ENV MAMBA_NO_BANNER=1
ENV PATH="${PATH}:/workspaces/morpheus/.devcontainer/bin"
