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
import shutil
import sys
import tarfile
import tempfile
import typing

import requests
import yaml

SCRIPT_DIR = os.path.relpath(os.path.dirname(__file__))
PROJ_ROOT = os.path.dirname(SCRIPT_DIR)

PIP_FLAGS_RE = re.compile(r"^--.*")
STRIP_VER_RE = re.compile(r"^([\w|-]+).*")
TAG_URL_PATH = "{base_url}/archive/refs/tags/{tag}.tar.gz"

# In some cases multiple packages are derived from a single upstream repo, please keep sorted
PACKAGE_ALIASES = {  # <conda package nanme>: <upstream name>
    "beautifulsoup4": "beautifulsoup",
    "elasticsearch": "elasticsearch-py",
    "grpcio": "grpc",
    "grpcio-status": "grpc",
    "milvus": "milvus-lite",
    "nlohmann_json": "json",
    'python': 'cpython',
    "python-confluent-kafka": "confluent-kafka-python",
    "python-graphviz": "graphviz",
    "torch": "pytorch",
}

KNOWN_GITHUB_URLS = {  # <package>: <github repo>, please keep sorted
    'appdirs': 'https://github.com/ActiveState/appdirs',
    'c-ares': 'https://github.com/c-ares/c-ares',
    'click': 'https://github.com/pallets/click',
    'confluent-kafka-python': 'https://github.com/confluentinc/confluent-kafka-python',
    'cpython': 'https://github.com/python/cpython',
    'cupy': 'https://github.com/cupy/cupy',
    'databricks-cli': 'https://github.com/databricks/databricks-cli',
    'datacompy': 'https://github.com/capitalone/datacompy',
    'dfencoder': 'https://github.com/AlliedToasters/dfencoder',
    'dill': 'https://github.com/uqfoundation/dill',
    'docker-py': 'https://github.com/docker/docker-py',
    'elasticsearch-py': 'https://github.com/elastic/elasticsearch-py',
    'feedparser': 'https://github.com/kurtmckee/feedparser',
    'graphviz': 'https://github.com/xflr6/graphviz',
    'grpc': 'https://github.com/grpc/grpc',
    'json': 'https://github.com/nlohmann/json',
    'librdkafka': 'https://github.com/confluentinc/librdkafka',
    'libwebp': 'https://github.com/webmproject/libwebp',
    'mlflow': 'https://github.com/mlflow/mlflow',
    'networkx': 'https://github.com/networkx/networkx',
    'numpydoc': 'https://github.com/numpy/numpydoc',
    'pip': 'https://github.com/pypa/pip',
    'pluggy': 'https://github.com/pytest-dev/pluggy',
    'protobuf': 'https://github.com/protocolbuffers/protobuf',
    'pybind11': 'https://github.com/pybind/pybind11',
    'pydantic': 'https://github.com/pydantic/pydantic',
    'pymilvus': 'https://github.com/milvus-io/pymilvus',
    'python-versioneer': 'https://github.com/python-versioneer/python-versioneer',
    'rapidjson': 'https://github.com/Tencent/rapidjson',
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
}

# Please keep sorted
KNOWN_FIRST_PARTY = {
    'cuda-cudart', 'cuda-nvrtc', 'cuda-nvtx', 'cuda-version', 'cudf', 'mrc', 'rapids-dask-dependency', 'tritonclient'
}

KNOWN_NON_CONDA_DEPS = [('dfencoder', '0.0.37')]

TAG_BARE = "{version}"
TAG_V_PREFIX = "v{version}"  # Default & most common tag format
TAG_NAME_DASH_BARE = "{name}-{version}"

GIT_TAG_FORMAT = {  # any packages not in this dict are assumned to have the TAG_V_PREFIX
    'appdirs': TAG_BARE,
    'click': TAG_BARE,
    'databricks-cli': TAG_BARE,
    'dill': TAG_NAME_DASH_BARE,
    'docker-py': TAG_BARE,
    'feedparser': TAG_BARE,
    'graphviz': TAG_BARE,
    'networkx': TAG_NAME_DASH_BARE,
    'pip': TAG_BARE,
    'pluggy': TAG_BARE,
    'python-versioneer': TAG_BARE,
    'scikit-learn': TAG_BARE,
    'sqlalchemy': lambda ver: f"rel_{ver.replace('.', '_')}",
    'websockets': TAG_BARE,
}

logger = logging.getLogger(__file__)


def mk_github_urls(packages: list[tuple[str, str]]) -> tuple[dict[str, typing.Any], list[str]]:
    matched = {}
    unmatched: list[str] = []
    for (pkg_name, pkg_version) in packages:
        if pkg_name in KNOWN_FIRST_PARTY:
            logger.debug("Skipping first party package: %s", pkg_name)
            continue

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

        tar_url = TAG_URL_PATH.format(base_url=repo_url, tag=tag)

        matched[github_name] = {'packages': [pkg_name], 'tag': tag, 'tar_url': tar_url}

    return (matched, unmatched)


def mk_request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response | None:
    try:
        response = session.request(method, url, allow_redirects=True, timeout=30)
        return response
    except requests.HTTPError as e:
        logger.error("Failed to fetch %s: %s", url, e)


def verify_tar_urls(session: requests.Session, dep_urls: dict[str, typing.Any]):
    github_names = sorted(dep_urls.keys())
    for github_name in github_names:
        dep_info = dep_urls[github_name]
        url = dep_info['tar_url']
        response = mk_request(session, "HEAD", url)

        is_valid = (response is not None and response.status_code == 200)
        dep_info['is_valid'] = is_valid

        msg = f"{github_name} : {dep_info['tag']} is_valid={is_valid}"
        if is_valid:
            logger.debug(msg)
        else:
            logger.error(msg)


def download_tars(session: requests.Session, dep_urls: dict[str, typing.Any], download_dir: str):
    github_names = sorted(dep_urls.keys())
    for github_name in github_names:
        dep_info = dep_urls[github_name]
        url = dep_info['tar_url']

        # When --skip_url_verify is set the is_valid key will not be present
        if dep_info.get('is_valid', True):
            tar_file = os.path.join(download_dir, f"{github_name}.tar.gz")
            if os.path.exists(tar_file) and tarfile.is_tarfile(tar_file):
                dep_info['tar_file'] = tar_file
                logger.info("Skipping download of %s, already exists: %s", github_name, tar_file)
                continue

            response = mk_request(session, "GET", url, stream=True)
            if (response is not None and response.status_code == 200):
                with open(tar_file, 'wb') as fh:
                    for chunk in response.iter_content(decode_unicode=False):
                        fh.write(chunk)

                dep_info['tar_file'] = tar_file
                logger.info("Downloaded %s: %s", github_name, tar_file)
            else:
                logger.error("Failed to fetch %s", url)
                continue
        else:
            logger.warning("Skipping download of invalid package %s", github_name)


def extract_tar_files(dep_urls: dict[str, typing.Any], extract_dir: str):
    github_names = sorted(dep_urls.keys())
    for github_name in github_names:
        dep_info = dep_urls[github_name]
        tar_file = dep_info.get('tar_file')
        if tar_file is None:
            logger.error("No tar file found for %s", github_name)
            continue

        if tarfile.is_tarfile(tar_file):
            with tarfile.open(tar_file, 'r:*') as tar:
                extract_location = os.path.join(extract_dir, github_name)
                tar.extractall(path=extract_location)
                logger.debug("Extracted %s: %s -> %s", github_name, tar_file, extract_location)
                dep_info['extract_location'] = extract_location
        else:
            logger.error("Not a valid tar file: %s", tar_file)


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


def merge_deps(declared_deps: list[str],
               other_deps: list[tuple[str, str]],
               resolved_conda_deps: dict[str, dict[str, typing.Any]]) -> list[tuple[str, str]]:
    merged_deps: list[tuple[str, str]] = other_deps.copy()
    for dep in declared_deps:
        # intentionally allow a KeyError to be raised in the case of an unmatched package
        pkg_info = resolved_conda_deps[dep]
        version = pkg_info['version'].split('+')[0]  # strip any conda variant info ex: 1.2.3+cuda11.0
        merged_deps.append((dep, version))

    # Return sorted list just for nicer debug output
    return sorted(merged_deps)


def print_summary(dep_urls: dict[str, typing.Any], unmatched_packages: list[str], download: bool,
                  extract: bool) -> list[str]:
    missing_packages = unmatched_packages.copy()

    if extract:
        check_key = 'extract_location'
    elif download:
        check_key = 'tar_file'
    else:
        check_key = 'is_valid'

    for pkg_name in sorted(dep_urls.keys()):
        pkg_info = dep_urls[pkg_name]
        if not pkg_info.get(check_key, False):
            missing_packages.append(pkg_name)

    if len(missing_packages) > 0:
        # Print summary of all packages that we couldn't find, and will need to be fetched manually
        logger.warning(
            "\n----------------------\n"
            "Packages that could not be downaloaded and will need to be fetched manually:\n%s",
            "\n".join(sorted(missing_packages)))

    return missing_packages


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

    argparser.add_argument('--skip_url_verify', default=False, action='store_true')
    argparser.add_argument('--download', default=False, action='store_true')

    argparser.add_argument('--download_dir',
                           default=None,
                           help="When --download is set, directory to download tar archives to, if unspecified, a "
                           "temporary directory will be created.")

    argparser.add_argument('--no_clean',
                           default=False,
                           action='store_true',
                           help="Do not remove temporary download directory.")

    argparser.add_argument('--extract', default=False, action='store_true')

    argparser.add_argument('--extract_dir',
                           default=None,
                           help="When --extract is set, directory to extract tar archives, if unspecified, a temporary "
                           "directory will be created.")

    argparser.add_argument("--log_level",
                           default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           help="Specify the logging level to use.")

    argparser.add_argument("--http_log_level",
                           default="WARNING",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           help="Specify the logging level to use for requests and urllib3.")

    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    log_level = logging._nameToLevel[args.log_level.upper()]
    logging.basicConfig(level=log_level, format="%(message)s")

    # Set the log level for requests and urllib3
    logging.getLogger('requests').setLevel(args.http_log_level)
    logging.getLogger("urllib3").setLevel(args.http_log_level)

    declared_deps = parse_env_file(args.conda_yaml)
    resolved_conda_deps = parse_json_deps(args.conda_json)

    merged_deps = merge_deps(declared_deps, KNOWN_NON_CONDA_DEPS, resolved_conda_deps)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Declared Yaml deps:\n%s", pprint.pformat(sorted(declared_deps)))
        logger.debug("Resolved Conda deps:\n%s", pprint.pformat(resolved_conda_deps))
        logger.debug("Merged deps:\n%s", pprint.pformat(merged_deps))

    (dep_urls, unmatched_packages) = mk_github_urls(merged_deps)
    if len(unmatched_packages) > 0:
        logger.error(
            "\n------------\nPackages without github info which will need to be fetched manually:\n%s\n------------\n",
            pprint.pformat(unmatched_packages))

    if not args.download and args.skip_url_verify:
        sys.exit(0)

    with requests.Session() as session:
        if not args.skip_url_verify:
            verify_tar_urls(session, dep_urls)

        download_dir: str | None = args.download_dir
        if args.download:
            if download_dir is None:
                download_dir = tempfile.mkdtemp(prefix="morpheus_deps_download_")
                logger.info("Created temporary download directory: %s", download_dir)

            download_tars(session, dep_urls, download_dir)

    extract_dir: str | None = args.extract_dir
    if args.extract:
        if extract_dir is None:
            extract_dir = tempfile.mkdtemp(prefix="morpheus_deps_extract_")
            logger.info("Created temporary extract directory: %s", extract_dir)

        extract_tar_files(dep_urls, extract_dir)

    if args.download_dir is None and download_dir is not None and not args.no_clean:
        logger.info("Removing temporary download directory: %s", download_dir)
        shutil.rmtree(download_dir)

    missing_packages = print_summary(dep_urls, unmatched_packages, args.download, args.extract)
    if args.extract:
        print(f"Exraction location: {extract_dir}")

    if len(missing_packages) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
