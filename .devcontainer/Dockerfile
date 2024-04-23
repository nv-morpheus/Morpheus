ARG BASE
ARG PYTHON_PACKAGE_MANAGER

FROM ${BASE} as base

# ===== install common packages ==================================================================

RUN <<EOF

set -x

apt-get update;

# apt-get install -y --no-install-recommends  \
#     libboost-{regex,system,filesystem}-dev  \
#     ;

rm -rf /tmp/*;
rm -rf /var/tmp/*;
rm -rf /var/cache/apt/*;
rm -rf /var/lib/apt/lists/*;

EOF

# ===== install node + camouflage ================================================================

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/include/node /usr/local/include/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node /opt/yarn-v*/bin/* /usr/local/bin/
COPY --from=node /opt/yarn-v*/lib/* /usr/local/lib/

RUN \
<<EOF

bash -c 'echo -e "
fund=false\n
audit=false\n
save-prefix=\n
--omit=optional\n
save-exact=true\n
package-lock=false\n
update-notifier=false\n
scripts-prepend-node-path=true\n
registry=https://registry.npmjs.org/\n
" | tee /root/.npmrc >/dev/null'

ln -s /usr/local/bin/node /usr/local/bin/nodejs
ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

echo "node version: $(node --version)"
echo " npm version: $(npm --version)"
echo "yarn version: $(yarn --version)"

npm install -g camouflage-server@0.15

EOF


FROM base as pip-base

ENV DEFAULT_VIRTUAL_ENV=morpheus

FROM base as conda-base

ENV DEFAULT_CONDA_ENV=morpheus

FROM ${PYTHON_PACKAGE_MANAGER}-base

ENV PROJECT_MANIFEST_YML="/home/coder/morpheus/manifest.yaml"
ENV PATH="${PATH}:/home/coder/morpheus/.devcontainer/bin"

ARG CUDA
ENV CUDAARCHS="RAPIDS"
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG RAPIDS
ENV RAPIDS=${RAPIDS}

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV VAULT_HOST="https://vault.ops.k8s.rapids.ai"
ENV HISTFILE="/home/coder/.cache/._bash_history"
