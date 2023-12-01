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
import sys

from setuptools import find_packages  # noqa: E402
from setuptools import setup  # noqa: E402

try:
    import versioneer
except ImportError:
    # we have a versioneer.py file living in the same directory as this file, but
    # if we're using pep 517/518 to build from pyproject.toml its not going to find it
    # https://github.com/python-versioneer/python-versioneer/issues/193#issue-408237852
    # make this work by adding this directory to the python path
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import versioneer

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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    author="NVIDIA Corporation",
    include_package_data=True,
    packages=find_packages(include=["morpheus*"], exclude=['tests']),
    install_requires=[
        # Only list the packages which cannot be installed via conda here.
        "pyarrow_hotfix",  # CVE-2023-47248. See morpheus/__init__.py for more details
    ],
    license="Apache",
    python_requires='>=3.10, <4',
    cmdclass=versioneer.get_cmdclass(),
    entry_points='''
        [console_scripts]
        morpheus=morpheus.cli:run_cli
        ''',
)
