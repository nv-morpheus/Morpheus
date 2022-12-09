FROM ghcr.io/trxcllnt/devcontainer-images/cmake-ninja-sccache-llvm-cuda-mambaforge:latest AS base

COPY bin/setup-morpheus-env.sh /opt/morpheus-devcontainer/bin/setup-morpheus-env

ENV MAMBA_NO_BANNER=1
ENV PATH="${PATH}:/opt/morpheus-devcontainer/bin"
