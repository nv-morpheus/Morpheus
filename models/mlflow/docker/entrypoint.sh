#!/bin/bash --login

# Activate the `mlflow` conda environment.
. /opt/conda/etc/profile.d/conda.sh
conda activate mlflow

# Source "source" file if it exists
SRC_FILE="/opt/docker/bin/entrypoint_source"
[ -f "${SRC_FILE}" ] && source "${SRC_FILE}"

# Run whatever the user wants.
exec "$@"