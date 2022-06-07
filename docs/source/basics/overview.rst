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

Overview
========

This section focuses on the Morpheus CLI and illustrates how the CLI can be used to configure and run a Morpheus
Pipeline.

Organization
------------

The Morpheus CLI is built on the Click Python package which allows for nested commands and chaining multiple commands
together. At a high level, the CLI is broken up into two main sections:

 * ``run``
    * For running NLP or FIL pipelines.
 * ``tools``
    * Tools/Utilities to help setup, configure and run pipelines and external resources

Users can get help on any command by passing ``--help`` to a command or sub-command. For example, to get help on the
tools:

.. code-block:: console

   $ morpheus tools --help
   Usage: morpheus tools [OPTIONS] COMMAND [ARGS]...

   Options:
     --help  Show this message and exit.  [default: False]

   Commands:
     autocomplete  Utility for installing/updating/removing shell completion for
                   Morpheus
     onnx-to-trt   Converts an ONNX model to a TRT engine

The help text will show arguments, options and all possible sub-commands. Help for each of these sub-commands can be
queried in the same manner:

.. code-block:: console

   $ morpheus tools onnx-to-trt --help
   Usage: morpheus tools onnx-to-trt [OPTIONS]

   Options:
     --input_model PATH              [required]
     --output_model PATH             [required]
     --batches <INTEGER INTEGER>...  [required]
     --seq_length INTEGER            [required]
     --max_workspace_size INTEGER    [default: 16000]
     --help                          Show this message and exit.  [default:
                                     False]

AutoComplete
------------

The Morpheus CLI supports bash, fish, zsh, and powershell autocompletion. To setup autocomplete, it must first be
installed. Morpheus comes with a tool to assist with this:

.. code-block:: console

   $ morpheus tools autocomplete install
   bash completion installed in ~/.bash_completion
   $ source ~/.bash_completion

After autocomplete has been installed, ``[TAB]`` can be used to show all commands, options and arguments when building
pipelines via the CLI:

.. code-block:: console

   $ morpheus run pipeline- # [TAB][TAB]
   pipeline-fil  pipeline-nlp
