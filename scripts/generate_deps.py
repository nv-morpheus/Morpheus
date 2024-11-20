#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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
"""
Script to automate downloading of source code for third party dependencies

Intentionally using as few third-party dependencies as possible to allow running this script outside of a Morpheus
Conda environment.
"""

import argparse
import json
import logging
import os
import pprint
import re
import sys
import typing

import yaml

SCRIPT_DIR = os.path.relpath(os.path.dirname(__file__))
PROJ_ROOT = os.path.dirname(SCRIPT_DIR)

PIP_FLAGS_RE = re.compile(r"^--.*")
STRIP_VER_RE = re.compile(r"^([\w|-]+).*")
TAG_URL_PATH = "{base_url}/releases/tag/{tag}"
TAG_URL_TAR_PATH = "{base_url}/archive/refs/tags/{tag}.tar.gz"

# In some cases multiple packages are derived from a single upstream repo
PACKAGE_ALIASES = {  # <conda package nanme>: <upstream name>
    "beautifulsoup4": "beautifulsoup",
    "elasticsearch": "elasticsearch-py",
    "grpcio": "grpc",
    "grpcio-status": "grpc",
    "milvus": "milvus-lite",
    "nlohmann_json": "json",
    "python-confluent-kafka": "confluent-kafka-python",
    "python-graphviz": "graphviz",
    "torch": "pytorch",
}

KNOWN_GITHUB_URLS = {  # <package>: <github repo>
    'c-ares': 'https://github.com/c-ares/c-ares',
    'click': 'https://github.com/pallets/click',
    'cpython': 'https://github.com/python/cpython',
    'databricks-cli': 'https://github.com/databricks/databricks-cli',
    'datacompy': 'https://github.com/capitalone/datacompy',
    'dill': 'https://github.com/uqfoundation/dill',
    'docker-py': 'https://github.com/docker/docker-py',
    'elasticsearch-py': 'https://github.com/elastic/elasticsearch-py',
    'feedparser': 'https://github.com/kurtmckee/feedparser',
    'grpc': 'https://github.com/grpc/grpc',
    'mlflow': 'https://github.com/mlflow/mlflow',
    'networkx': 'https://github.com/networkx/networkx',
    'json': 'https://github.com/nlohmann/json',
    'numpydoc': 'https://github.com/numpy/numpydoc',
    'pip': 'https://github.com/pypa/pip',
    'pluggy': 'https://github.com/pytest-dev/pluggy',
    'protobuf': 'https://github.com/protocolbuffers/protobuf',
    'pybind11': 'https://github.com/pybind/pybind11',
    'pydantic': 'https://github.com/pydantic/pydantic',
    'pymilvus': 'https://github.com/milvus-io/pymilvus',
    'confluent-kafka-python': 'https://github.com/confluentinc/confluent-kafka-python',
    'graphviz': 'https://github.com/xflr6/graphviz',
    'rapidjson': 'https://github.com/Tencent/rapidjson',
    'librdkafka': 'https://github.com/confluentinc/librdkafka',
    'rdma-core': 'https://github.com/linux-rdma/rdma-core',
    'requests': 'https://github.com/psf/requests',
    'requests-cache': 'https://github.com/requests-cache/requests-cache',
    'RxCpp': 'https://github.com/ReactiveX/RxCpp',
    'scikit-learn': 'https://github.com/scikit-learn/scikit-learn',
    'sqlalchemy': 'https://github.com/sqlalchemy/sqlalchemy',
    'pytorch': 'https://github.com/pytorch/pytorch',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'typing_utils': 'https://github.com/bojiang/typing_utils',
    'watchdog': 'https://github.com/gorakhargosh/watchdog',
    'websockets': 'https://github.com/python-websockets/websockets',
    'python-versioneer': 'https://github.com/python-versioneer/python-versioneer',
    'dfencoder': 'https://github.com/AlliedToasters/dfencoder'
}

TAG_BARE = "{version}"
TAG_V_PREFIX = "v{version}"  # Default & most common tag format
TAG_NAME_DASH_BARE = "{name}-{version}"

GIT_TAG_FORMAT = {  # any packages not in this dict are assumned to have the TAG_V_PREFIX
    'click': TAG_BARE,
    'databricks-cli': TAG_BARE,
    'dill': TAG_NAME_DASH_BARE,
    'docker-py': TAG_BARE,
    'feedparser': TAG_BARE,
    'networkx': TAG_NAME_DASH_BARE,
    'pip': TAG_BARE,
    'pluggy': TAG_BARE,
    'graphviz': TAG_BARE,
    'scikit-learn': TAG_BARE,
    'sqlalchemy': lambda ver: f"rel_{ver.replace('.', '_')}",
    'websockets': TAG_BARE,
    'python-versioneer': TAG_BARE,
    'dfencoder': TAG_V_PREFIX
}

logger = logging.getLogger()


def mk_github_urls(packages: list[tuple[str, str]]) -> dict[str, typing.Any]:
    matched = {}
    unmatched: list[str] = []
    for (pkg_name, pkg_version) in packages:
        github_name = PACKAGE_ALIASES.get(pkg_name, pkg_name)
        if github_name != pkg_name:
            logger.debug("Package %s is knwon as %s", pkg_name, github_name)

        # Some packages share a single upstream repo
        if github_name in matched:
            matched[github_name]['packages'].append(pkg_name)
            continue

        try:
            repo_url = KNOWN_GITHUB_URLS[github_name]
        except KeyError:
            unmatched.append(pkg_name)
            continue

        tag_formatter = GIT_TAG_FORMAT.get(github_name, TAG_V_PREFIX)
        if isinstance(tag_formatter, str):
            tag = tag_formatter.format(name=github_name, version=pkg_version)
        else:
            tag = tag_formatter(pkg_version)

        tag_url = TAG_URL_PATH.format(base_url=repo_url, tag=tag)
        tag_tar_url = TAG_URL_TAR_PATH.format(base_url=repo_url, tag=tag)

        matched[github_name] = {'packages': [pkg_name], 'tag_url': tag_url, 'tag_tar_url': tag_tar_url}

    return {"matched": matched, "unmatched": unmatched}


def parse_json_deps(json_file: str) -> dict[str, dict[str, typing.Any]]:
    with open(json_file, 'r', encoding="utf-8") as f:
        json_data = json.load(f)

    # Create a new dict keyed by package name
    packages = {pkg['name']: pkg for pkg in json_data}
    return packages


def strip_version(dep: str) -> str:
    match = STRIP_VER_RE.match(dep)
    if match is not None:
        return match.group(1)

    logger.error("Failed to strip version from dependency: %s", dep)
    sys.exit(1)


def parse_dep(parsed_deps: set, dep: str):
    pkg_name = strip_version(dep)
    if pkg_name in parsed_deps:
        logger.error("Duplicate package found: %s", pkg_name)
        sys.exit(1)

    parsed_deps.add(pkg_name)


def parse_env_file(yaml_env_file: str) -> list[str]:
    with open(yaml_env_file, 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    deps = yaml_data['dependencies']

    parsed_deps = set()
    pip_deps = []
    for dep in deps:
        if isinstance(dep, dict):
            if len(dep) == 1 and 'pip' in dep:
                pip_deps.extend(dep['pip'])
            else:
                logger.error("Unsupported dependency format: %s", dep)
                sys.exit(1)
        else:
            parse_dep(parsed_deps, dep)

    for dep in pip_deps:
        if PIP_FLAGS_RE.match(dep) is None:  # skip pip arguments
            parse_dep(parsed_deps, dep)

    # Return sorted list just for nicer debug output
    return sorted(parsed_deps)


def merge_deps(declared_deps: list[str], resolved_conda_deps: dict[str, dict[str,
                                                                             typing.Any]]) -> list[tuple[str, str]]:
    merged_deps: list[tuple[str, str]] = []
    for dep in declared_deps:
        # intentionally allow a KeyError to be raised in the case of an unmatched package
        pkg_info = resolved_conda_deps[dep]
        merged_deps.append((dep, pkg_info['version']))

    # Return sorted list just for nicer debug output
    return sorted(merged_deps)


def parse_args():
    argparser = argparse.ArgumentParser(
        "Download source code for third party dependencies specified in a Conda environment yaml file, by default "
        "unless --download is specified only the github URLs will be printed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('--conda_yaml',
                           default=os.path.join(PROJ_ROOT, "conda/environments/runtime_cuda-125_arch-x86_64.yaml"),
                           help=("Conda environment file to read dependencies from"))

    argparser.add_argument('--conda_json',
                           default=os.path.join(PROJ_ROOT, ".tmp/container_pkgs.json"),
                           help=("JSON formatted output of the resolved Conda environment. Generated by running: "
                                 "`./docker/run_container_release.sh conda list --json > .tmp/container_pkgs.json` "
                                 "This is used to determine the exact version number actually used by a package which "
                                 "specifies a version range in the Conda environment file."))

    argparser.add_argument('--skip_verify', default=False, action='store_true')
    argparser.add_argument('--download', default=False, action='store_true')

    argparser.add_argument("--log_level",
                           default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           help="Specify the logging level to use.")

    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    log_level = logging._nameToLevel[args.log_level.upper()]
    logging.basicConfig(level=log_level, format="%(message)s")

    declared_deps = parse_env_file(args.conda_yaml)
    resolved_conda_deps = parse_json_deps(args.conda_json)

    merged_deps = merge_deps(declared_deps, resolved_conda_deps)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Declared Yaml deps:\n%s", pprint.pformat(sorted(declared_deps)))
        logger.debug("Resolved Conda deps:\n%s", pprint.pformat(resolved_conda_deps))
        logger.debug("Merged deps:\n%s", pprint.pformat(merged_deps))

    github_urls = mk_github_urls(merged_deps)
    unmatched_packages = github_urls['unmatched']
    if len(unmatched_packages) > 0:
        logger.error(
            "\n------------\nPackages without github info which will need to be fetched manually:\n%s\n------------\n",
            pprint.pformat(unmatched_packages))

    if not args.skip_verify:
        pass


if __name__ == "__main__":
    main()
