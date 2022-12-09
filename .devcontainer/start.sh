#!/bin/sh

conda_env_find(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

MORPHEUS_DEFAULT_CONDA_ENV=${MORPHEUS_DEFAULT_CONDA_ENV:-morpheus}

conda_env_update(){onda
    mamba env update -n ${MORPHEUS_DEFAULT_CONDA_ENV} -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda11.5_dev.yml
}

if conda_env_find "${MORPHEUS_DEFAULT_CONDA_ENV}" ; then
    conda activate ${MORPHEUS_DEFAULT_CONDA_ENV}
else
    read -p "Morpheus conda environment not found. Would you like to create it? [yn] " yn
    case $yn in
        [Yy]* ) mamba env create --name ${MORPHUES_DEFAULT_CONDA_ENV} -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda11.5_dev.yml && conda activate ${env_name};;
    esac
fi
