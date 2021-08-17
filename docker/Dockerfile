ARG FROM_IMAGE=gpuci/miniconda-cuda
ARG CUDA_VER=11.2
ARG LINUX_VER=ubuntu18.04
FROM ${FROM_IMAGE}:${CUDA_VER}-runtime-${LINUX_VER} AS base

# Required arguments
ARG IMAGE_TYPE=base
ARG RAPIDS_CHANNEL=rapidsai-nightly
ARG RAPIDS_VER=21.06
ARG PYTHON_VER=3.8

# The TensorRT Version must be the full version (i.e. 7.2.2.3) and match Triton EXACTLY
ARG TENSORRT_VERSION=7.2.2.3

# Capture argument used for FROM
ARG CUDA_VER

# Install build-essential. Needed to call `make`
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# All code will be under /workspace
WORKDIR /workspace

RUN conda config --set ssl_verify false \
    && /opt/conda/bin/conda install -y -n base -c conda-forge mamba && conda clean -afy

COPY docker/conda/environments/dev_cuda${CUDA_VER}.yml ./docker/conda/environments/

RUN /opt/conda/bin/mamba env create -n morpheus \
  --file docker/conda/environments/dev_cuda${CUDA_VER}.yml &&\
  conda clean -afy && \
  sed -i 's/conda activate base/conda activate morpheus/g' ~/.bashrc

# Enables all RUN commands to use the conda env
SHELL ["/opt/conda/bin/conda", "run", "-n", "morpheus", "/bin/bash", "-c"]

# Install nvidia-pyindex which is needed for tritonclient[all] but must be
# installed before setup is run. Also force reinstall of just streamz since it
# gets installed as a dependency of custreamz. The arguments `--upgrade
# --no-deps --force-reinstall` are required to force reinstalling just streamz
# without all its dependencies
RUN pip install nvidia-pyindex && \
  pip install nvidia-tensorrt==${TENSORRT_VERSION} && \
  pip install --upgrade --no-deps --force-reinstall git+https://github.com/mdemoret-nv/streamz.git@async#egg=streamz


# Copy the source
COPY . ./

# Build/Install morpheus and put CLI on PATH
RUN pip install .

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
ENTRYPOINT [ "/usr/bin/tini", "--", "docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# Use morpheus by default
CMD [ "morpheus" ]
