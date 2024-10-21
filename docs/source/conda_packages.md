<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Morpheus Conda Packages
The Morpheus stages are the building blocks for creating pipelines. The stages are organized into libraries by use case. The current libraries are:
- `morpheus-core`
- `morpheus-dfp`
- `morpheus-llm`

The  libraries are hosted as Conda packages on the [`nvidia`](https://anaconda.org/nvidia/) channel.

The split into multiple libraries allows for a more modular approach to using the Morpheus stages. For example, if you are building an application for Digital Finger Printing, you can install just the `morpheus-dfp` library. This reduces the size of the installed package. It also limits the dependencies eliminating unnecessary version conflicts.


## Morpheus Core
The `morpheus-core` library contains the core stages that are common across all use cases. The Morpheus core library is built from the source code in the `python/morpheus` directory of the Morpheus repository. The core library is installed as a dependency when you install any of the other Morpheus libraries.
To set up a Conda environment with the [`morpheus-core`](https://anaconda.org/nvidia/morpheus-core) library you can run the following commands:
### Create a Conda environment
```bash
export CONDA_ENV_NAME=morpheus
conda create -n $CONDA_ENV_NAME python=3.10
conda activate $CONDA_ENV_NAME
```
### Add Conda channels
These channel are required for installing the runtime dependencies
```bash
conda config --env --add channels conda-forge &&\
  conda config --env --add channels nvidia &&\
  conda config --env --add channels rapidsai &&\
  conda config --env --add channels pytorch
```
### Install the `morpheus-core` library
```bash
conda install -c nvidia morpheus-core
```
The `morpheus-core` Conda package installs the `morpheus` python package. It also pulls down all the necessary Conda runtime dependencies for the core stages including [`mrc`](https://anaconda.org/nvidia/mrc) and [`libmrc`](https://anaconda.org/nvidia/libmrc).
### Install additional `Pypi` dependencies
Some of the stages in the core library require additional dependencies that are hosted on `Pypi`. These dependencies are included as a requirements file in the `morpheus` python package. The requirements files can be located and installed by running the following command:
```bash
python3 <<EOF
import importlib.resources
import subprocess
requirements_file = importlib.resources.path("morpheus", "requirements_morpheus_core.txt")
subprocess.call(f"pip install -r {requirements_file}".split())
EOF
```

## Morpheus DFP
Digital Finger Printing (DFP) is a technique used to identify anomalous behavior and uncover potential threats in the environment​. The `morpheus-dfp` library contains stages for DFP. It is built from the source code in the `python/morpheus_dfp` directory of the Morpheus repository. To set up a Conda environment with the [`morpheus-dfp`](https://anaconda.org/nvidia/morpheus-dfp) library you can run the following commands:
### Create a Conda environment
```bash
export CONDA_ENV_NAME=morpheus-dfp
conda create -n $CONDA_ENV_NAME python=3.10
conda activate $CONDA_ENV_NAME
```
### Add Conda channels
These channel are required for installing the runtime dependencies
```bash
conda config --env --add channels conda-forge &&\
  conda config --env --add channels nvidia &&\
  conda config --env --add channels rapidsai &&\
  conda config --env --add channels pytorch
```
### Install the `morpheus-dfp` library
```bash
conda install -c nvidia morpheus-dfp
```
The `morpheus-dfp` Conda package installs the `morpheus_dfp` python package. It also pulls down all the necessary Conda runtime dependencies including [`morpheus-core`](https://anaconda.org/nvidia/morpheus-core).
### Install additional `Pypi` dependencies
Some of the DFP stages in the library require additional dependencies that are hosted on `Pypi`. These dependencies are included as a requirements file in the `morpheus_dfp` python package. And can be installed by running the following command:
```bash
python3 <<EOF
import importlib.resources
import subprocess
requirements_file = importlib.resources.path("morpheus_dfp", "requirements_morpheus_dfp.txt")
subprocess.call(f"pip install -r {requirements_file}".split())
EOF
```

## Morpheus LLM
The `morpheus-llm` library contains stages for Large Language Models (LLM) and  Vector Databases. These stages are used for setting up Retrieval Augmented Generation (RAG) pipelines. The `morpheus-llm` library is built from the source code in the `python/morpheus_llm` directory of the Morpheus repository.
To set up a Conda environment with the [`morpheus-llm`](https://anaconda.org/nvidia/morpheus-dfp) library you can run the following commands:
### Create a Conda environment
```bash
export CONDA_ENV_NAME=morpheus-llm
conda create -n $CONDA_ENV_NAME python=3.10
conda activate $CONDA_ENV_NAME
```
### Add Conda channels
These channel are required for installing the runtime dependencies
```bash
conda config --env --add channels conda-forge &&\
  conda config --env --add channels nvidia &&\
  conda config --env --add channels rapidsai &&\
  conda config --env --add channels pytorch
```
### Install the `morpheus-llm` library
```bash
conda install -c nvidia morpheus-llm
```
The `morpheus-llm` Conda package installs the `morpheus_llm` python package. It also pulls down all the necessary Conda packages including [`morpheus-core`](https://anaconda.org/nvidia/morpheus-core).
### Install additional `Pypi` dependencies
Some of the stages in the library require additional dependencies that are hosted on `Pypi`. These dependencies are included as a requirements file in the `morpheus_llm` python package. And can be installed by running the following command:
```bash
python3 <<EOF
import importlib.resources
import subprocess
requirements_file = importlib.resources.path("morpheus_llm", "requirements_morpheus_llm.txt")
subprocess.call(f"pip install -r {requirements_file}".split())
EOF
```

## Miscellaneous
### Morpheus Examples
The Morpheus examples are not included in the Morpheus Conda packages. To use them you need to clone the Morpheus repository and run the examples from source. For details refer to the [Morpheus Examples](./examples.md).

### Namespace Updates
If you were using a Morpheus release prior to 24.10 you may need to update the namespace for the DFP, LLM and vector database stages.

A script, `scripts/morpheus_namespace_update.py`, has been provide to help with that and can be run as follows:
```bash
python scripts/morpheus_namespace_update.py --directory <directory> --dfp
```
```bash
python scripts/morpheus_namespace_update.py --directory <directory> --llm
```
