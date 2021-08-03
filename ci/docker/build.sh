#!/bin/bash
set -e

# Overwrite HOME to WORKSPACE
export HOME="$WORKSPACE"

# Install gpuCI tools
curl -s https://raw.githubusercontent.com/rapidsai/gpuci-tools/main/install.sh | bash
source ~/.bashrc
cd ~

# Show env
gpuci_logger "Exposing current environment..."
env

# Select dockerfile based on matrix var
DOCKERFILE="docker/Dockerfile"
DOCKER_CONTEXT="."
gpuci_logger "Using Dockerfile: ${DOCKERFILE}"
gpuci_logger "Using Context: ${DOCKER_CONTEXT}"

# Debug output selected dockerfile
gpuci_logger ">>>> BEGIN Dockerfile <<<<"
cat ${DOCKERFILE}
gpuci_logger ">>>> END Dockerfile <<<<"

# Get build info ready
gpuci_logger "Preparing build config..."
BUILD_TAG="cuda${CUDA_VER}-${LINUX_VER}-py${PYTHON_VER}"

# Setup initial BUILD_ARGS
BUILD_ARGS="--no-cache \
  --build-arg CUDA_VER=${CUDA_VER} \
  --build-arg LINUX_VER=${LINUX_VER} \
  --build-arg PYTHON_VER=${PYTHON_VER}"

# Ouput build config
gpuci_logger "Build config info..."
echo "Build args: ${BUILD_ARGS}"
gpuci_logger "Docker build command..."
echo "docker build --pull -t morpheus:${BUILD_TAG} ${BUILD_ARGS} -f ${DOCKERFILE} ${DOCKER_CONTEXT}"

# Build image
gpuci_logger "Starting build..."
docker build --pull -t morpheus:${BUILD_TAG} ${BUILD_ARGS} -f ${DOCKERFILE} ${DOCKER_CONTEXT}

# List image info
gpuci_logger "Displaying image info..."
docker images morpheus:${BUILD_TAG}
