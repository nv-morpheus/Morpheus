ARG MORPHEUS_CONTAINER=nvcr.io/nvidia/morpheus/morpheus
ARG MORPHEUS_CONTAINER_VERSION=v22.08.00-runtime

FROM ${MORPHEUS_CONTAINER}:${MORPHEUS_CONTAINER_VERSION} as base

# Fix the entrypoint to work with different WORKDIR
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

SHELL ["/bin/bash", "-c"]

# Install vault
RUN apt-get update \
    && apt-get install -y \
      apt-utils \
      jq \
      lsb-release \
      software-properties-common \
    && curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add - \
    && apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
    && apt-get update \
    && apt-get install --reinstall -y  \
      vault \
    && apt-get clean all \
    && rm -rf /var/lib/apt/lists/*

# Fix vault install per: https://github.com/hashicorp/vault/issues/10924#issuecomment-1197259930
RUN setcap -r /usr/bin/vault

# # Install NGC CLI
# COPY ${UBA_COMMON}/utils/*.sh ./
# RUN chmod +x *.sh \
#   && bash ngc-cli-install.sh

# Install DFP dependencies
RUN source activate morpheus \
    && mamba install -y -c conda-forge \
        boto3 \
        dill \
        ipywidgets \
        jupyterlab \
        kfp \
        librdkafka \
        mlflow \
        nb_conda_kernels \
        papermill \
        s3fs

WORKDIR /work/examples/dfp_workflow/morpheus

# # This will get used by pipelines for the --s3_cache option
# ENV DFP_S3_CACHE="/work/examples/dfp_workflow/morpheus/.s3_cache"

# Set the tracking URI for mlflow
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

# Copy the sources
COPY . ./

CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
