ARG FROM_IMAGE=gpuci/miniforge-cuda
ARG CUDA_VER=11.4
ARG LINUX_VER=ubuntu20.04
ARG TRITON_VER=r22.02
FROM ${FROM_IMAGE}:${CUDA_VER}-runtime-${LINUX_VER} AS base

WORKDIR /mlflow

COPY . ./

# need to do an upgrade in case gpuci is stale for whatever reason
RUN apt-get update && apt upgrade -y && apt-get install -y procps vim
RUN /opt/conda/bin/conda env create -f docker/conda/mlflow-env.yml
RUN sed -i 's/conda activate base/conda activate mlflow/g' ~/.bashrc

SHELL ["/opt/conda/bin/conda", "run", "-n", "mlflow", "/bin/bash", "-c"]

ARG TRITON_DIR=/mlflow/triton-inference-server
ARG TRITON_VER=r21.12

RUN mkdir ${TRITON_DIR} && \
    cd ${TRITON_DIR} && \
    git clone -b ${TRITON_VER} --depth 1 https://github.com/triton-inference-server/server && \
    source activate mlflow && \
    cd ${TRITON_DIR}/server/deploy/mlflow-triton-plugin && \
    pip install .

RUN ln -sf ${TRITON_DIR}/server/deploy/mlflow-triton-plugin/scripts/publish_model_to_mlflow.py /mlflow && \
    mkdir /mlflow/artifacts

ENV MLFLOW_MODEL_REPO=/mlflow/artifacts
ENV MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow-db.sqlite

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "docker/entrypoint.sh" ]

CMD [ "/bin/bash" ]
