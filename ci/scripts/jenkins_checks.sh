#!/bin/bash

set -e

source ci/scripts/jenkins_common.sh

gpuci_logger "Installing CI dependencies"
mamba install -q -y -c conda-forge "yapf=0.32.0"

gpuci_logger "Runing Python style checks"
ci/scripts/python_checks.sh

gpuci_logger "Checking copyright headers"
python ci/scripts/copyright.py --verify-apache-v2 --git-diff-commits "origin/${CHANGE_TARGET}" ${GIT_COMMIT}
