# Contributing to Morpheus

Contributions to Morpheus fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/NVIDIA/Morpheus/issues/new)
    describing in detail the problem or new feature. The Morpheus team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, please comment on the issue to notify the
    team.
2. To propose and implement a new Feature, please file a new feature request
    [issue](https://github.com/NVIDIA/Morpheus/issues/new). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, please ask in a comment.

As contributors and maintainers to this project,
you are expected to abide by Morpheus' code of conduct.
More information can be found at: [Contributor Code of Conduct](CODE_OF_CONDUCT.md).

## Code contributions

### Your first issue

1. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/NVIDIA/Morpheus/issues).
2. Comment on the issue stating that you are going to work on it.
3. Code! Make sure to update unit tests! Ensure the [license headers are set properly](#Licensing).
4. When done, [create your merge request](https://github.com/NVIDIA/Morpheus/compare).
5. Wait for other developers to review your code and update code as needed.
6. Once reviewed and approved, a Morpheus developer will merge your merge request.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues for our next release in our [project boards](https://github.com/NVIDIA/Morpheus/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where Morpheus developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable
contributing. Start with _Step 2_ above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the MR.

## Setting Up Your Build Environment

The following instructions are for developers who are getting started with the Morpheus repository. The Morpheus development environment is flexible (Docker, Conda and bare metal workflows) but has a high number of dependencies that can be difficult to setup. These instructions outline the steps for setting up a development environment inside a Docker container or on a host machine with Conda.

All of the following instructions assume several variables have been set:
 - `MORPHEUS_ROOT`: The Morpheus repository has been checked out at a location specified by this variable. Any non-absolute paths are relative to `MORPHEUS_ROOT`.
 - `PYTHON_VER`: The desired Python version. Minimum required is 3.8
 - `RAPIDS_VER`: The desired RAPIDS version for all RAPIDS libraries including cuDF and RMM. This is also used for Triton.
 - `CUDA_VER`: The desired CUDA version to use.

### Clone the repository and pull large file data from Git LFS

```bash
MORPHEUS_HOME=$(pwd)/morpheus
git clone https://github.com/NVIDIA/Morpheus.git $MORPHEUS_HOME
cd $MORPHEUS_HOME
```
The large model and data files in this repo are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/). These files will be required for running the training/validation scripts and example pipelines for the Morpheus pre-trained models.

If `Git LFS` is not installed before cloning the repository, the large files will not be pulled. If this is the case, follow the instructions for installing `Git LFS` from [here](https://git-lfs.github.com/), and then run the following command.
```bash
git lfs pull
```

### Build in Docker Container

This workflow utilizes a docker container to set up most dependencies ensuring a consistent environment.

#### Prerequisites

- Pascal architecture or better
- NVIDIA driver `450.80.02` or higher
- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

1. Build the development container
   ```bash
   ./docker/build_container_dev.sh
   ```
   1. The container tag will default to `morpheus:YYMMDD` where `YYMMDD` is the current 2 digit year, month and day respectively. The tag can be overridden by setting `DOCKER_IMAGE_TAG`. For example,
      ```bash
      DOCKER_IMAGE_TAG=my_tag ./docker/build_container_dev.sh
      ```
      Would build the container `morpheus:my_tag`.
   1. Note: This does not build any Morpheus or Neo code and defers building the code until the entire repo can be mounted into a running container. This allows for faster incremental builds during development.
2. Set up `ssh-agent` to allow container to pull from private repos
   ```bash
   eval `ssh-agent -s`
   ssh-add
   ```
3. Run the development container
   ```bash
   ./docker/run_container_dev.sh
   ```
   1. The container tag follows the same rules as `build_container_dev.sh` and will default to the current `YYMMDD`. Specify the desired tag with `DOCKER_IMAGE_TAG`. i.e. `DOCKER_IMAGE_TAG=my_tag ./docker/run_container_dev.sh`
   2. This will automatically mount the current working directory to `/workspace`.
   3. Some of the validation tests require launching a triton docker container within the morpheus container. To enable this you will need to grant the morpheus contrainer access to your host OS's docker socket file with:
      ```bash
      DOCKER_EXTRA_ARGS="-v /var/run/docker.sock:/var/run/docker.sock" ./docker/run_container_dev.sh
      ```
      Then once the container is started you will need to install some extra packages to enable launching docker containers:
      ```bash
      ./docker/install_docker.sh
      ```

4. Compile Morpheus
   ```bash
   ./scripts/compile.sh
   ```
   This script will run both CMake Configure with default options and CMake build.
5. Install Morpheus
   ```bash
   pip install -e /workspace
   ```
   Once Morpheus has been built, it can be installed into the current virtual environment.
6. [Run Morpheus](./README.md#running-morpheus)
   ```bash
   morpheus run pipeline-nlp ...
   ```
   At this point, Morpheus can be fully used. Any changes to Python code will not require a rebuild. Changes to C++ code will require calling `./scripts/compile.sh`. Installing Morpheus is only required once per virtual environment.

### Build in a Conda Environment

If a Conda environment on the host machine is preferred over Docker, it is relatively easy to install the necessary dependencies (In reality, the Docker workflow creates a Conda environment inside the container).

Note: These instructions assume the user is using `mamba` instead of `conda` since it's improved solver speed is very helpful when working with a large number of dependencies. If you are not familiar with `mamba` you can install it with `conda install -n base -c conda-forge mamba` (Make sure to only install into the base environment). `mamba` is a drop in replacement for `conda` and all conda commands are compatible between the two.

#### Prerequisites

- Pascal architecture or better
- NVIDIA driver `450.80.02` or higher
- [CUDA 11.0+](https://developer.nvidia.com/cuda-downloads)
- `conda` or `mamba`
  - See the [Getting Started Guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) if `conda` is not already installed
  - [Optional] Install `mamba` to speed up the package solver (highly recommended):

      ```bash
      conda activate base
      conda install -c conda-forge mamba
      ```

  - **Note:** `mamba` should only be installed once in the base environment

1. Create a new Conda environment
   ```bash
   export CUDAToolkit_ROOT=/usr/local/cuda-{CUDA_VER}
   mamba env create -n morpheus -f ./docker/conda/environments/cuda${CUDA_VER}_dev.yml
   conda activate morpheus
   ```
   This creates an environment named `morpheus` with all necessary dependencies, and activates that environment.
2. Build cuDF
   ```bash
   # Clone cuDF
   git clone -b branch-${RAPIDS_VER} --depth 1 https://github.com/rapidsai/cudf ${MORPHEUS_ROOT}/.cache/cudf
   cd ${MORPHEUS_ROOT}/.cache/cudf
   # Apply the Morpheus cuDF patch
   git apply --whitespace=fix ${MORPHEUS_ROOT}/cmake/deps/patches/cudf.patch
   # Build cuDF libraries
   ./build.sh --ptds libcudf cudf libcudf_kafka cudf_kafka
   cd ${MORPHEUS_ROOT}
   ```
   This will checkout, patch, build and install cuDF with the necessary fixes to allow Morpheus to work smoothly with cuDF DataFrames in C++.
3. Build Morpheus
   ```bash
   ./scripts/compile.sh
   ```
   This script will run both CMake Configure with default options and CMake build.
4. Install Morpheus
   ```bash
   pip install -e ${MORPHEUS_ROOT}
   ```
   Once Morpheus has been built, it can be installed into the current virtual environment.
5. Install camouflage, needed for the unittests
   ```bash
   npm install -g camouflage-server
   ```
6. Run Morpheus
   ```bash
   morpheus run pipeline-nlp ...
   ```
   At this point, Morpheus can be fully used. Any changes to Python code will not require a rebuild. Changes to C++ code will require calling `./scripts/compile.sh`. Installing Morpheus is only required once per virtual environment.

### Troubleshooting the Build

Due to the large number of dependencies, it's common to run into build issues. The follow are some common issues, tips, and suggestions:

 - Issues with the build cache
   - To avoid rebuilding every compilation unit for all dependencies after each change, a fair amount of the build is cached. By default, the cache is located at `${MORPHEUS_ROOT}/.cache`. The cache contains both compiled object files, source repositories, ccache files, clangd files and even the cuDF build.
   - The entire cache folder can be deleted at any time and will be redownload/recreated on the next build
 - Message indicating `git apply ...` failed
   - Many of the dependencies require small patches to make them work. These patches must be applied once and only once. If you see this error, try deleting the offending package from the `build/_deps/<offending_packag>` directory or from `.cache/cpm/<offending_package>`.
   - If all else fails, delete the entire `build/` directory and `.cache/` directory.

## Licensing
Morpheus is licensed under the Apache v2.0 license. All new source files including CMake and other build scripts should contain the Apache v2.0 license header. Any edits to existing source code should update the date range of the copyright to the current year. The format for the license header is:

```
/*
 * SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 ```

### Thirdparty code
Thirdparty code included in the source tree (that is not pulled in as an external dependency) must be compatible with the Apache v2.0 license and should retain the original license along with a url to the source. If this code is modified, it should contain both the Apache v2.0 license followed by the original license of the code and the url to the original code.

Ex:
```
/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Original Source: https://github.com/org/other_project
//
// Original License:
// ...
```


---

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md \
Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst
