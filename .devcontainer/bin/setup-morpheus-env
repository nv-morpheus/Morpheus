#!/bin/sh

conda_env_find(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

ENV_NAME=${ENV_NAME:-morpheus}

if conda_env_find "${ENV_NAME}" ; \
then mamba env update --name ${ENV_NAME} -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda11.5_dev.yml; \
else mamba env create --name ${ENV_NAME} -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda11.5_dev.yml; \
fi

sed -ri "s/conda activate base/conda activate $ENV_NAME/g" ~/.bashrc;
