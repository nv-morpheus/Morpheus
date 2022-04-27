# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import logging
import os
import subprocess
import time

import pytest
import requests


def pytest_addoption(parser: pytest.Parser):
    """
    Adds command line options for running specfic tests that are disabled by default
    """
    parser.addoption(
        "--run_slow",
        action="store_true",
        dest="run_slow",
        help="Run slow tests that would otherwise be skipped",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    This function will add parameterizations for the `config` fixture depending on what types of config the test
    supports
    """

    # Only care about the config fixture
    if ("config" not in metafunc.fixturenames):
        return

    use_cpp = metafunc.definition.get_closest_marker("use_cpp") is not None
    use_python = metafunc.definition.get_closest_marker("use_python") is not None

    if (use_cpp and use_python):
        raise RuntimeError(
            "Both markers (use_cpp and use_python) were added to function {}. Remove markers to support both.".format(
                metafunc.definition.nodeid))
    elif (not use_cpp and not use_python):
        # Add the markers to the parameters
        metafunc.parametrize("config",
                             [
                                 pytest.param(True, marks=pytest.mark.use_cpp, id="use_cpp"),
                                 pytest.param(False, marks=pytest.mark.use_python, id="use_python")
                             ],
                             indirect=True)


def pytest_runtest_setup(item):
    if (not item.config.getoption("--run_slow")):
        if (item.get_closest_marker("slow") is not None):
            pytest.skip("Skipping slow tests by default. Use --run_slow to enable")


def pytest_collection_modifyitems(items):
    """
    To support old unittest style tests, try to determine the mark from the name
    """
    for item in items:
        if "no_cpp" in item.nodeid:
            item.add_marker(pytest.mark.use_python)
        elif "cpp" in item.nodeid:
            item.add_marker(pytest.mark.use_cpp)


@pytest.fixture(scope="function")
def config_only_cpp():
    """
    Use this fixture in unittest style tests to indicate a lack of support for C++. Use via
    `@pytest.mark.usefixtures("config_only_cpp")`
    """

    from morpheus.config import Config
    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(True)

    yield Config()


@pytest.fixture(scope="function")
def config_no_cpp():
    """
    Use this fixture in unittest style tests to indicate support for C++. Use via
    `@pytest.mark.usefixtures("config_no_cpp")`
    """

    from morpheus.config import Config
    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(False)

    yield Config()


@pytest.fixture(scope="function")
def config(request: pytest.FixtureRequest):
    """
    For new pytest style tests, get the config by using this fixture. It will setup the config based on the marks set on
    the object. If no marks are added to the test, it will be parameterized for both C++ and python. For example,

    ```
    @pytest.mark.use_python
    def my_python_test(config: Config):
        ...
    ```
    """

    from morpheus.config import Config
    from morpheus.config import CppConfig

    if (not hasattr(request, "param")):
        use_cpp = request.node.get_closest_marker("use_cpp") is not None
        use_python = request.node.get_closest_marker("use_python") is not None

        assert use_cpp != use_python, "Invalid config"

        CppConfig.set_should_use_cpp(True if use_cpp else False)

    else:
        CppConfig.set_should_use_cpp(True if request.param else False)

    yield Config()


@pytest.fixture(scope="function")
def restore_environ():
    orig_vars = os.environ.copy()
    yield os.environ

    # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
    for key in list(os.environ.keys()):
        orig_val = orig_vars.get(key)
        if orig_val is not None:
            os.environ[key] = orig_val
        else:
            del (os.environ[key])


@pytest.fixture(scope="function")
def reload_modules(request: pytest.FixtureRequest):
    marker = request.node.get_closest_marker("reload_modules")
    yield

    if marker is not None:
        modules = marker.args[0]
        if not isinstance(modules, list):
            modules = [modules]

        for mod in modules:
            importlib.reload(mod)


@pytest.fixture(scope="function")
def chdir_tmpdir(request: pytest.FixtureRequest, tmp_path):
    """
    Executes a test in the tmp_path directory
    """
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


def wait_for_camouflage(popen, root_dir, host="localhost", port=8000, timeout=5):
    ready = False
    elapsed_time = 0.0
    sleep_time = 0.1
    url = "http://{}:{}/ping".format(host, port)
    while not ready and elapsed_time < timeout and popen.poll() is None:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                ready = r.json()['message'] == 'I am alive.'
        except Exception:
            pass

        if not ready:
            time.sleep(sleep_time)
            elapsed_time += sleep_time

    if popen.poll() is not None:
        raise RuntimeError("camouflage server exited with status code={} details in: {}".format(
            popen.poll(), os.path.join(root_dir, 'camouflage.log')))

    return ready


@pytest.fixture(scope="function")
def launch_mock_triton():
    """
    Launches a mock triton server using camouflage (https://testinggospels.github.io/camouflage/) with a package
    rooted at `root_dir` and configured with `config`.

    This function will wait for up to `timeout` seconds for camoflauge to startup

    This function is a no-op if the `MORPHEUS_NO_LAUNCH_CAMOUFLAGE` environment variable is defined, which can
    be useful during test development to run camouflage by hand.
    """
    from utils import TEST_DIRS

    root_dir = TEST_DIRS.mock_triton_servers_dir
    startup_timeout = 5
    shutdown_timeout = 1

    launch_camouflage = os.environ.get('MORPHEUS_NO_LAUNCH_CAMOUFLAGE') is None
    if launch_camouflage:
        popen = subprocess.Popen(["camouflage", "--config", "config.yml"],
                                 cwd=root_dir,
                                 stderr=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL)

        logging.info("Launching camouflage in {} with pid: {}".format(root_dir, popen.pid))

        if startup_timeout > 0:
            if not wait_for_camouflage(popen, root_dir, timeout=startup_timeout):
                raise RuntimeError("Failed to launch camouflage server")

        yield

        logging.info("killing pid {}".format(popen.pid))

        elapsed_time = 0.0
        sleep_time = 0.1
        stopped = False

        # It takes a little while to shutdown
        while not stopped and elapsed_time < shutdown_timeout:
            popen.kill()
            stopped = (popen.poll() is not None)
            if not stopped:
                time.sleep(sleep_time)
                elapsed_time += sleep_time

    else:
        yield
