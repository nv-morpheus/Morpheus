set +x

echo "DOCA_BUILD_ID       : ${DOCA_BUILD_ID}"
echo "DOCA_VERSION        : ${DOCA_VERSION}"
echo "DPDK_VERSION        : ${DPDK_VERSION}"
echo "DOCA_REPO_HOST      : ${DOCA_REPO_HOST}"
echo "DOCA_ARTIFACTS_HOST : ${DOCA_ARTIFACTS_HOST}"

mkdir -p ./docker/deb && \
wget -O ./docker/deb/doca-host-repo-ubuntu2204_1.6.0-0.0.9-221109-160039-daily.1.6.0023.1.5.8.1.0.1.1_amd64.deb \
    https://${DOCA_REPO_HOST}/doca-repo-2.2.0/doca-repo-2.2.0-0.0.1-230405-143032-daily/doca-host-repo-ubuntu2204_2.2.0-0.0.1-230405-143032-daily.2.0.2004.2devflexio.23.04.0.2.3.0_amd64.deb && \
wget -O ./docker/deb/doca-apps-dev_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-apps-dev_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-apps_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-apps_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-grpc-dev_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-grpc-dev_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-grpc_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-grpc_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-libs_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-libs_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-prime-runtime_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-prime-runtime_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-prime-sdk_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-prime-sdk_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-prime-tools_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-prime-tools_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/doca-samples_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-samples_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/libdoca-libs-dev_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/doca-services_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/libdoca-libs-dev_${DOCA_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/libdoca-libs-dev_${DOCA_VERSION}_amd64.deb

wget -O ./docker/deb/mlnx-dpdk-dev_${DPDK_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/mlnx-dpdk-dev_${DPDK_VERSION}_amd64.deb

wget -O ./docker/deb/mlnx-dpdk-doc_${DPDK_VERSION}_all.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/mlnx-dpdk-doc_${DPDK_VERSION}_all.deb

wget -O ./docker/deb/mlnx-dpdk_${DPDK_VERSION}_amd64.deb \
    https://${DOCA_ARTIFACTS_HOST}/doca-gpunet/${DOCA_BUILD_ID}/doca-gpu-mlnx-dpdk/mlnx-dpdk_${DPDK_VERSION}_amd64.deb

dpkg -i ./docker/deb/doca-host-repo*.deb;

apt-get update
apt-get install -y libjson-c-dev meson cmake pkg-config
apt-get install -y ./docker/deb/mlnx-dpdk*.deb
apt-get install -y ./docker/deb/*doca*.deb