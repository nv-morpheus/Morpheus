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

echo "Installing Protobuf"

PROTOBUF_VERSION=${PROTOBUFVERSION:-latest}

if [ $PROTOBUF_VERSION == latest ]; then
    check_packages jq;
    PROTOBUF_VERSION="$(wget -O- -q https://api.github.com/repos/protocolbuffers/protobuf/releases/latest | jq -r ".tag_name" | tr -d 'v')";
fi

tmpdir="/tmp/protobuf";
mkdir -p ${tmpdir}
git clone \
    --branch v${PROTOBUF_VERSION} \
    --single-branch \
    --depth 1 \
    --recurse-submodules \
    --shallow-submodules \
    -j$(nproc --ignore=2) \
    https://github.com/protocolbuffers/protobuf.git ${tmpdir} \
    && cd ${tmpdir}

cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON.
cmake --build . --parallel $(nproc --ignore=2)
cmake --install .

rm -rf /var/tmp/* \
       /var/cache/apt/* \
       /var/lib/apt/lists/* \
       ${tmpdir};
