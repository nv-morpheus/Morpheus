
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
      brings the world’s best AI computing to communication networks.
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

   $ docker pull nvcr.io/ea-nvidia-morpheus/morpheus-sdk-cli

Launch an interactive container to start using Morpheus:

.. code-block:: console

   $ docker run --rm -ti --gpus=all morpheus bash
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

See :doc:`basic_usage` for more information on using the CLI.

Building Local Image
^^^^^^^^^^^^^^^^^^^^

Building the image locally is best suited for users who prefer working within a Docker container, want to avoid installing many dependencies or have a moderate amount of customization. This method requires pulling the source code and manually building the container and does not require the user to setup a Conda environment and install dependencies. Users can use either the CLI or Python interface.

Prerequisites
"""""""""""""
 * `Docker <https://docs.docker.com/get-docker/>`__
 * `The NVIDIA container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`__

To get started, first clone the Morpheus repo:

.. code-block:: bash

   # Make sure to recurse the submodules
   git clone --recurse-submodules ssh://git@gitlab-master.nvidia.com:12051/morpheus/morpheus.git
   # Change directory to the repo root
   cd morpheus

.. note::

   Cloning the repo may take a while to download large data objects and models.

To build the container:

.. code-block:: bash

   docker build -t morpheus -f docker/Dockerfile .

From this point, follow the previous getting started section for running the CLI.

.. _outside-of-a-container:

Outside of a Container
^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This is not the preferred way to use Morpheus. Morpheus requires a large amount of dependencies and this method should only be used by advanced and experienced users

Running Morpheus outside of a container requires the most setup, but offers the most flexibility and customization. Users of this method will need the source code and will be required to install several dependencies in a Conda virtual environment.

Prerequisites
"""""""""""""
 * `Conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`__
 * `CUDA <https://developer.nvidia.com/cuda-toolkit>`__
    * While CUDA can be installed with Conda, it requires installing the matching CUDA SDK outside of the Conda environment.

To get started, first clone the Morpheus repo:

.. code-block:: bash

   # Make sure to recurse the submodules
   git clone --recurse-submodules ssh://git@gitlab-master.nvidia.com:12051/morpheus/morpheus.git
   # Change directory to the repo root
   cd morpheus

.. note::

   Cloning the repo may take a while to download large data objects and models.

Next, create a Conda environment and install the necessary dependencies.

.. code-block:: bash

   conda create -n morpheus -c conda-forge python=${PYTHON_VER}
   conda activate morpheus
   conda install -c rapidsai \
      -c nvidia \
      -c conda-forge \
      -c defaults \
      cudatoolkit=${CUDA_VER} \
      cudf_kafka=${RAPIDS_VER} \
      python=${PYTHON_VER}

   # Installing nvidia-pyindex is required for Triton integration
   pip install nvidia-pyindex

Where ``$PYTHON_VER``, ``$CUDA_VER``, and ``$RAPIDS_VER`` represent the desired Python version, CUDA version and, RAPIDS
version, respectively. Finally, build Morpheus:

.. code-block:: bash

   pip install .
   # Or for a debug/editable installation
   pip install -e .

See :doc:`basic_usage` for more information on using the CLI and :doc:`advanced_usage` for more information on using the Python API.

.. toctree::
   :maxdepth: 20
   :caption: Contents:
   :hidden:

   basic_usage
   advanced_usage
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
