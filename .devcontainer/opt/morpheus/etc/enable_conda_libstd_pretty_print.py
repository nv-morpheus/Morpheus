# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- python -*-
import os
import subprocess
import sys

import gdb

conda_env_path = os.environ.get("CONDA_PREFIX", None)

if (conda_env_path is not None):

    gcc_path = os.environ.get("GCC", None)

    if (gcc_path is None):
        print(
            "Could not find gcc from $GCC: '{}'. Ensure gxx_linux-64, gcc_linux-64, sysroot_linux-64, and gdb have been installed into the current conda environment"
            .format(gcc_path))
    else:
        # Get the GCC version
        result = subprocess.run([gcc_path, '-dumpversion'], stdout=subprocess.PIPE)
        gcc_version = result.stdout.decode("utf-8").strip()

        # Build the gcc python path
        gcc_python_path = os.path.join(conda_env_path, "share", "gcc-{}".format(gcc_version), "python")

        if (os.path.exists(gcc_python_path)):

            # Add to the path for the pretty printer
            sys.path.insert(0, gcc_python_path)

            # Now register the pretty printers
            from libstdcxx.v6 import register_libstdcxx_printers
            register_libstdcxx_printers(gdb.current_objfile())

            print("Loaded stdlibc++ pretty printers")
        else:
            print("Could not find gcc python files at: {}".format(gcc_python_path))
            print(
                "Ensure gxx_linux-64, gcc_linux-64, sysroot_linux-64, and gdb have been installed into the current conda environment"
            )
