..
   SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


.. This role is needed at the index to set the default backtick role
.. role:: py(code)
   :language: python
   :class: highlight

Welcome to Morpheus documentation!
==================================

NVIDIA Morpheus is an open AI application framework that provides cybersecurity developers with a highly optimized AI
pipeline and pre-trained AI capabilities that, for the first time, allow them to instantaneously inspect all IP traffic
across their data center fabric. Bringing a new level of security to data centers, Morpheus provides dynamic protection,
real-time telemetry, adaptive policies, and cyber defenses for detecting and remediating cybersecurity threats.

Features
--------

 * Built on RAPIDS
    * Built on the RAPIDS™ libraries, deep learning frameworks, and NVIDIA Triton™ Inference Server, Morpheus simplifies
      the analysis of logs and telemetry to help detect and mitigate security threats.
 * AI Cybersecurity Capabilities
    * Deploy your own models using common deep learning frameworks. Or get a jump-start in building applications to
      identify leaked sensitive information, detect malware, and identify errors via logs by using one of NVIDIA’s
      pre-trained and tested models.
 * Real-Time Telemetry
    * Morpheus can receive rich, real-time network telemetry from every NVIDIA® BlueField® DPU-accelerated server in the
      data center without impacting performance. Integrating the framework into a third-party cybersecurity offering
      brings the world's best AI computing to communication networks.
 * DPU-Connected
    * The NVIDIA BlueField Data Processing Unit (DPU) can be used as a telemetry agent for receiving critical data
      center communications into Morpheus. As an optional addition to Morpheus, BlueField DPU also extends static
      security logging to a sophisticated dynamic real-time telemetry model that evolves with new policies and threat
      intelligence.

Getting Started
---------------

The best way to get started with Morpheus will vary depending on the goal of the user. These users fall into two large groups: those who want to use the pre-built pipelines exactly as they are with few modifications (see :ref:`using-ngc-container`) and those who want to use Morpheus as a framework for implementing their own end-to-end workflows (see :ref:`outside-of-a-container`).

.. _using-ngc-container:

Using NGC Container
^^^^^^^^^^^^^^^^^^^

Accessing Morpheus by pulling the pre-built NGC container is best suited for users who do not need any customization and
are only interested in running Morpheus via the CLI. The pre-built container does not require checking out the source
code and is best suited for users who are new to Morpheus and don't require any customization.

Prerequisites
"""""""""""""
 * `Docker <https://docs.docker.com/get-docker/>`__
 * `The NVIDIA container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`__

To get started, first pull the NGC container:

.. code-block:: console

   $ docker pull nvcr.io/nvidia/morpheus/morpheus:22.06-runtime

Launch an interactive container to start using Morpheus:

.. code-block:: console

   $ docker run --rm -ti --net=host --gpus=all nvcr.io/nvidia/morpheus/morpheus:22.06-runtime bash
   (morpheus) root@958a683a8a26:/workspace# morpheus --help
   Usage: morpheus [OPTIONS] COMMAND [ARGS]...Options:
     --debug / --no-debug            [default: False]
     --log_level [CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG]
                                     Specify the logging level to use.  [default:
                                     WARNING]

     --log_config_file FILE          Config file to use to configure logging. Use
                                     only for advanced situations. Can accept
                                     both JSON and ini style configurations

     --version                       Show the version and exit.  [default: False]
     --help                          Show this message and exit.  [default:
                                     False]


   Commands:
     run    Run one of the available pipelines
     tools  Run a utility tool

See :doc:`basics/overview` for more information on using the CLI.

Building Localy
---------------
To get started, first clone the Morpheus repo:

.. code-block:: bash

   export PYTHON_VER=3.8
   export RAPIDS_VER=22.06
   export CUDA_VER=11.5
   export MORPHEUS_ROOT=$(pwd)/morpheus

   git clone https://github.com/NVIDIA/Morpheus.git morpheus
   # Change directory to the repo root
   cd morpheus

Where ``$PYTHON_VER``, ``$CUDA_VER``, and ``$RAPIDS_VER`` represent the desired Python version, CUDA version and, RAPIDS
version, respectively.

.. note::
The Morpheus contains several large  model and data files in this repo are stored using `Git Large File Storage (LFS) <https://git-lfs.github.com/>`__. These files will be required for running the training/validation scripts and example pipelines for the Morpheus pre-trained models.

By default only those files stored in LFS strictly needed for running Morpheus are included when the Morpheus repository is cloned. Additional datasets can be downloaded using the `scripts/fetch_data.py` script. Usage of the script is as follows:

.. code-block:: bash

   scripts/fetch_data.py fetch <dataset> [<dataset>...]


At time of writing the defined datasets are:
 * all - Metaset includes all others
 * examples - Data needed by scripts in the `examples` subdir
 * models - Morpheus models (largest dataset)
 * tests - Data used by unittests
 * validation - Subset of the models dataset needed by some unittests

To download just the examples and models:

.. code-block:: bash

   scripts/fetch_data.py fetch examples models


To download the data needed for unittests:

.. code-block:: bash

   scripts/fetch_data.py fetch tests validation


If `Git LFS` is not installed before cloning the repository, the large files will not be pulled. If this is the case, follow the instructions for installing `Git LFS` from `here <https://git-lfs.github.com/>`__, and then run the following command.

.. code-block:: bash

   scripts/fetch_data.py fetch all

From this point, follow the instructions in either the :ref:`building-local-image<Building Local Image>` or :ref:`outside-of-a-container<Outside of a Container>` section.

.. building-local-image:

Building Local Image
^^^^^^^^^^^^^^^^^^^^

Building the image locally is best suited for users who prefer working within a
Docker container, want to avoid installing many dependencies or have a moderate
amount of customization. This method requires pulling the source code and
manually building the container and does not require the user to setup a Conda
environment and install dependencies. Users can use either the CLI or Python
interface.

Prerequisites
"""""""""""""
 * `Docker <https://docs.docker.com/get-docker/>`__
 * `The NVIDIA container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`__


To build the container:

.. code-block:: bash

   ./docker/build_container_dev.sh

To run the development container:

.. code-block:: bash

   ./docker/run_container_dev.sh

From this point, follow the previous getting started section for running the CLI.

.. _outside-of-a-container:

Outside of a Container
^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This is not the preferred way to use Morpheus. Morpheus requires a large
   amount of dependencies and this method should only be used by advanced and
   experienced users only.

Running Morpheus outside of a container requires the most setup, but offers the
most flexibility and customization. Users of this method will need the source
code and will be required to install several dependencies in a Conda virtual
environment.

Prerequisites
"""""""""""""
 * `Conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`__
 * `Mamba <https://github.com/mamba-org/mamba>`__
    * Once ``conda`` is installed, ``mamba`` can be instaled with ``conda install -n base -c conda-forge mamba`` (Make sure to only install into the base environment)
 * `CUDA <https://developer.nvidia.com/cuda-toolkit>`__
    * While CUDA can be installed with Conda, it requires installing the matching CUDA SDK outside of the Conda environment.

Create a Conda environment and install the necessary dependencies.

.. code-block:: bash

   conda install -c conda-forge mamba
   mamba env create -f ./docker/conda/environments/cuda${CUDA_VER}_dev.yml
   conda activate morpheus

Next, build Morpheus:

.. code-block:: bash

   ./scripts/compile.sh

   pip install .
   # Or for a debug/editable installation
   pip install -e .

See :doc:`basics/overview` for more information on using the CLI.


.. toctree::
   :maxdepth: 20
   :hidden:

   morpheus_quickstart_guide

.. toctree::
   :caption: Basic Usage via CLI
   :maxdepth: 20
   :hidden:

   basics/overview
   basics/building_a_pipeline
   basics/examples

.. toctree::
   :caption: Developer Guide:
   :maxdepth: 20
   :hidden:

   developer_guide/architecture
   developer_guide/guides
   api

.. toctree::
   :maxdepth: 20
   :caption: Extra Information:
   :hidden:

   extra_info/performance
   extra_info/troubleshooting
   extra_info/known_issues

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
