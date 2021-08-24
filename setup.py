# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See the docstring in versioneer.py for instructions. Note that you must
# re-run 'versioneer.py setup' after changing this section, and commit the
# resulting files.

from setuptools import setup, find_packages

import versioneer

setup(
    name="morpheus",
    version=versioneer.get_version(),
    description="Morpheus",
    classifiers=[
        "Development Status :: 3 - Alpha",

        # Utilizes NVIDIA GPUs
        "Environment :: GPU :: NVIDIA CUDA",

        # Audience (TODO: (MDD) Audit these)
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",

        # License
        "License :: OSI Approved :: Apache Software License",

        # Only support Python 3.8+
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    author="NVIDIA Corporation",
    packages=find_packages(include=["morpheus", "morpheus.*"]),
    include_package_data=True,
    install_requires=[
        "streamz @ git+https://github.com/mdemoret-nv/streamz.git@on_completed#egg=streamz",
        "appdirs",
        "click-completion",
        "click<8",
        "configargparse",
        "docker",
        "grpcio-channelz",
        "networkx",
        "tqdm",
        "tritonclient[all]",
        "typing-utils",
        "xgboost",
    ],
    license="Apache",
    python_requires='>=3.8, <4',
    cmdclass=versioneer.get_cmdclass(),
    entry_points='''
        [console_scripts]
        morpheus=morpheus.cli:run_cli
      ''',
)
