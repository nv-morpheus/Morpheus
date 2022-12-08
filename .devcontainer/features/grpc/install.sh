#! /usr/bin/env bash
set -ex

apt_get_update()
{
    if [ "$(find /var/lib/apt/lists/* | wc -l)" = "0" ]; then
        echo "Running apt-get update..."
        apt-get update -y;
    fi
}

# Checks if packages are installed and installs them if not
check_packages() {
    if ! dpkg -s "$@" > /dev/null 2>&1; then
        apt_get_update
        echo "Installing prerequisites: $@";
        DEBIAN_FRONTEND=noninteractive \
        apt-get -y install --no-install-recommends "$@"
    fi
}

check_packages build-essential autoconf libtool pkg-config

echo "Installing gRPC"

GRPC_VERSION=${GRPCVERSION:-latest}

if [ $GRPC_VERSION == latest ]; then
    check_packages jq;
    GRPC_VERSION="$(wget -O- -q https://api.github.com/repos/grpc/grpc/releases/latest | jq -r ".tag_name" | tr -d 'v')";
fi

tmpdir="/tmp/grpc";
mkdir -p ${tmpdir}
git clone \
    --branch v${GRPC_VERSION} \
    --single-branch \
    --depth 1 \
    --recurse-submodules \
    --shallow-submodules \
    -j$(nproc --ignore=2) https://github.com/grpc/grpc.git ${tmpdir} \
    && cd ${tmpdir}


cmake -S . -B build -GNinja \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_CSHARP_EXT=OFF \
    -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF \
    -DgRPC_PROTOBUF_PROVIDER=package

make -j$(nproc --ignore=2)
cmake --build build --target install

rm -rf /var/tmp/* \
       /var/cache/apt/* \
       /var/lib/apt/lists/* \
       ${tmpdir};
