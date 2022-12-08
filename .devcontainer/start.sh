#!/bin/sh

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

env_name=morpheus

if find_in_conda_env "${env_name}" ; then
    conda activate ${env_name}
else
    read -p "Morpheus conda environment not found. Would you like to create it? [yn] " yn
    case $yn in
        [Yy]* ) mamba env create --name ${env_name} -f /workspaces/morpheus/docker/conda/environments/cuda11.5_dev.yml && conda activate ${env_name};;
    esac
fi
