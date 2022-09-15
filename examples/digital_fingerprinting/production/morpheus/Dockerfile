ARG MORPHEUS_CONTAINER=nvcr.io/nvidia/morpheus/morpheus
ARG MORPHEUS_CONTAINER_VERSION=v22.09.00-runtime
ARG ADDITIONAL_GROUPS
ARG FIXUID_VERSION=0.5.1

FROM ${MORPHEUS_CONTAINER}:${MORPHEUS_CONTAINER_VERSION} as common_base

# Install fixuid
RUN curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz | tar -C /usr/bin -xzf - && \
    chown root:root /usr/bin/fixuid && chmod 4755 /usr/bin/fixuid && mkdir -p /etc/fixuid && \
    bash -c 'echo -e "\
user: morpheus\n\
group: morpheus\n\
paths:\n\
  - /home/morpheus\n\
  - /opt/conda/envs/morpheus\n\
" | tee /etc/fixuid/config.yml >/dev/null' && \
    \
    # Add a non-root user
    useradd \
      --uid 1000 --shell /bin/bash \
      --user-group ${ADDITIONAL_GROUPS} \
      --create-home --home-dir /home/morpheus \
      morpheus

# Fix the entrypoint to work with different WORKDIR
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "fixuid", "-q", "/workspace/docker/entrypoint.sh" ]

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

# Copy the conda_env file
COPY ./conda_env.yml ./

# Install DFP dependencies
RUN source activate morpheus \
    && mamba env update -n morpheus -f ./conda_env.yml

USER morpheus

FROM common_base as base

WORKDIR /work/examples/dfp_workflow/morpheus

# This will get used by pipelines for the --s3_cache option
# ENV DFP_S3_CACHE="/work/examples/dfp_workflow/morpheus/.s3_cache"

# Copy the sources
COPY . ./

CMD ["bash", "-c", "./launch.sh"]
