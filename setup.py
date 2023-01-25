# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import os

# Required to install torch via setup.py
# Note: this is order dependent
os.environ["PIP_FIND_LINKS"] = "https://download.pytorch.org/whl/cu113/torch_stable.html"

import versioneer  # noqa: E402
from setuptools import find_packages  # noqa: E402
from setuptools import setup  # noqa: E402

setup(
    name="morpheus",
    version=versioneer.get_version(),
    description="Morpheus",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    author="NVIDIA Corporation",
    include_package_data=True,
    packages=find_packages(include=["morpheus", "morpheus.*"], exclude=['tests']),
    install_requires=[
        # Only list the packages which cannot be installed via conda here. Should mach the requirements in
        # docker/conda/environments/requirements.txt
        "dfencoder @ git+https://github.com/nv-morpheus/dfencoder.git@branch-23.01#egg=dfencoder",
        "torch==1.12.0+cu113",
        "tritonclient[all]==2.17.*",  # Force to 2.17 since they require grpcio==1.41 for newer versions
    ],
    license="Apache",
    python_requires='>=3.8, <4',
    cmdclass=versioneer.get_cmdclass(),
    entry_points='''
        [console_scripts]
        morpheus=morpheus.cli:run_cli
        ''',
)
