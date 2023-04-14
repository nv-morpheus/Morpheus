# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import importlib
import logging
import os
import signal
import subprocess
import sys
import time
import typing
import warnings
from collections import namedtuple
from functools import partial

import pytest
import requests

# actual topic names not important, but we will need two of them.
KAFKA_TOPICS = namedtuple('KAFKA_TOPICS', ['input_topic', 'output_topic'])('morpheus_input_topic',
                                                                           'morpheus_output_topic')

zookeeper_proc = None
kafka_server = None
kafka_consumer = None
pytest_kafka_setup_error = None


def init_pytest_kafka():
    """
    pytest_kafka is currently required to be installed manually, along with a download of Kafka and a functional JDK.
    Since the Kafka tests don't run by default, we will silently fail to initialize unless --run_kafka is enabled.

    Issue #9 should make the instalation of Kafka simpler:
    https://gitlab.com/karolinepauls/pytest-kafka/-/issues/9
    """
    global zookeeper_proc, kafka_server, kafka_consumer, pytest_kafka_setup_error
    try:
        import pytest_kafka
        os.environ['KAFKA_OPTS'] = "-Djava.net.preferIPv4Stack=True"
        # Initialize pytest_kafka fixtures following the recomendations in:
        # https://gitlab.com/karolinepauls/pytest-kafka/-/blob/master/README.rst
        KAFKA_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(pytest_kafka.__file__)), 'kafka/bin/')
        KAFKA_BIN = os.path.join(KAFKA_SCRIPTS, 'kafka-server-start.sh')
        ZOOKEEPER_BIN = os.path.join(KAFKA_SCRIPTS, 'zookeeper-server-start.sh')

        for kafka_script in (KAFKA_BIN, ZOOKEEPER_BIN):
            if not os.path.exists(kafka_script):
                raise RuntimeError("Required Kafka script not found: {}".format(kafka_script))

        teardown_fn = partial(pytest_kafka.terminate, signal_fn=subprocess.Popen.kill)
        zookeeper_proc = pytest_kafka.make_zookeeper_process(ZOOKEEPER_BIN, teardown_fn=teardown_fn)
        kafka_server = pytest_kafka.make_kafka_server(KAFKA_BIN, 'zookeeper_proc', teardown_fn=teardown_fn)
        kafka_consumer = pytest_kafka.make_kafka_consumer('kafka_server',
                                                          group_id='morpheus_unittest_reader',
                                                          client_id='morpheus_unittest_reader',
                                                          seek_to_beginning=True,
                                                          kafka_topics=[KAFKA_TOPICS.output_topic])

        return True
    except Exception as e:
        pytest_kafka_setup_error = e
        return False


pytest_kafka_avail = init_pytest_kafka()


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

    parser.addoption(
        "--run_kafka",
        action="store_true",
        dest="run_kafka",
        help="Run kafka tests that would otherwise be skipped",
    )

    parser.addoption(
        "--run_benchmark",
        action="store_true",
        dest="run_benchmark",
        help="Run benchmark tests that would otherwise be skipped",
    )

    parser.addoption(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"],
        dest="log_level",
        help="A specific log level to use during testing. Defaults to WARNING if not set.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    This function will add parameterizations for the `config` fixture depending on what types of config the test
    supports
    """

    # === use_cpp Parameterize ===
    use_cpp = metafunc.definition.get_closest_marker("use_cpp") is not None
    use_python = metafunc.definition.get_closest_marker("use_python") is not None

    use_cpp_param = pytest.param(True, marks=pytest.mark.use_cpp(added_by="generate_tests"), id="use_cpp")
    use_python_param = pytest.param(False, marks=pytest.mark.use_python(added_by="generate_tests"), id="use_python")

    _set_use_cpp_params = []

    if ("use_cpp" in metafunc.fixturenames):
        # Need to add some params since the fixture was requested

        # Add cpp unless use_cpp == True and use_python == False
        if not (use_python and not use_cpp):
            _set_use_cpp_params.append(use_cpp_param)

        # Add python unless use_cpp == False and use_python == True
        if not (not use_python and use_cpp):
            _set_use_cpp_params.append(use_python_param)

    elif (use_cpp and use_python):
        # Need to parameterize since we have multiple
        _set_use_cpp_params.extend([use_cpp_param, use_python_param])

    if (len(_set_use_cpp_params) > 0):
        metafunc.parametrize("_set_use_cpp", _set_use_cpp_params, indirect=True)

    # === df_type Parameterize ===
    if ("df_type" in metafunc.fixturenames):
        # df_type fixture was requested. Only parameterize if both marks or neither marks are found. Otherwise, the
        # fixture will determine it from the mark
        use_cudf = metafunc.definition.get_closest_marker("use_cudf") is not None
        use_pandas = metafunc.definition.get_closest_marker("use_pandas") is not None

        if (use_pandas == use_cudf):
            metafunc.parametrize(
                "df_type",
                [
                    pytest.param("cudf", marks=pytest.mark.use_cudf(added_by="generate_tests"), id="use_cudf"),
                    pytest.param("pandas", marks=pytest.mark.use_pandas(added_by="generate_tests"), id="use_pandas")
                ],
                indirect=True)


def pytest_runtest_setup(item):
    if (not item.config.getoption("--run_slow")):
        if (item.get_closest_marker("slow") is not None):
            pytest.skip("Skipping slow tests by default. Use --run_slow to enable")

    if (not item.config.getoption("--run_kafka")):
        if (item.get_closest_marker("kafka") is not None):
            pytest.skip("Skipping Kafka tests by default. Use --run_kafka to enable")

    if (not item.config.getoption("--run_benchmark")):
        if (item.get_closest_marker("benchmark") is not None):
            pytest.skip("Skipping benchmark tests by default. Use --run_benchmark to enable")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: typing.List[pytest.Item]):
    """
    To support old unittest style tests, try to determine the mark from the name
    """

    if config.getoption("--run_kafka") and not pytest_kafka_avail:
        raise RuntimeError(
            "--run_kafka requested but pytest_kafka not available due to: {}".format(pytest_kafka_setup_error))

    for item in items:
        if "no_cpp" in item.nodeid and item.get_closest_marker("use_python") is None:
            item.add_marker(pytest.mark.use_python(added_in="collection_modifyitems"))
        elif "cpp" in item.nodeid and item.get_closest_marker("use_cpp") is None:
            item.add_marker(pytest.mark.use_cpp(added_in="collection_modifyitems"))

    def should_filter_test(item: pytest.Item):

        use_cpp = item.get_closest_marker("use_cpp")
        use_pandas = item.get_closest_marker("use_pandas")

        if (use_cpp and use_pandas):
            return False

        return True

    # Now delete tests with incompatible markers
    items[:] = [x for x in items if should_filter_test(x)]


def clear_handlers(logger):
    handlers = logger.handlers.copy()
    for h in handlers:
        logger.removeHandler(h)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    clear_handlers(logging.getLogger("morpheus"))
    clear_handlers(logging.getLogger())


# This fixture will be used by all tests.
@pytest.fixture(scope="function", autouse=True)
def _set_use_cpp(request: pytest.FixtureRequest):

    do_use_cpp: bool = True

    # Check for the param if this was indirectly set
    if (hasattr(request, "param") and isinstance(request.param, bool)):
        do_use_cpp = request.param
    else:
        # If not, check for the marker and use that
        use_cpp = request.node.get_closest_marker("use_cpp") is not None
        use_python = request.node.get_closest_marker("use_python") is not None

        if (use_cpp and use_python):
            raise RuntimeError(
                "Both markers (use_cpp and use_python) were added to function {}. Remove markers to support both.".
                format(request.node.nodeid))
        else:
            # This will default to True or follow use_cpp
            do_use_cpp = not use_python

    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(do_use_cpp)

    yield do_use_cpp


# This fixture will be used by all tests.
@pytest.fixture(scope="function")
def use_cpp(_set_use_cpp: bool):

    # Just return the set value
    yield _set_use_cpp


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
def df_type(request: pytest.FixtureRequest):

    df_type_str: typing.Literal["cudf", "pandas"]

    # Check for the param if this was indirectly set
    if (hasattr(request, "param")):
        assert request.param in ["pandas", "cudf"], "Invalid parameter for df_type"

        df_type_str = request.param
    else:
        # If not, check for the marker and use that
        use_pandas = request.node.get_closest_marker("use_pandas") is not None
        use_cudf = request.node.get_closest_marker("use_cudf") is not None

        if (use_pandas and use_cudf):
            raise RuntimeError(
                "Both markers (use_cpp and use_python) were added to function {}. Remove markers to support both.".
                format(request.node.nodeid))
        else:
            # This will default to "cudf" or follow use_pandas
            df_type_str = "cudf" if not use_pandas else "pandas"

    yield df_type_str


@pytest.fixture(scope="function")
def config(use_cpp: bool):
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

    yield Config()


@pytest.fixture(scope="function")
def kafka_topics():
    """
    Returns a named tuple of Kafka topic names in the form of (input_topic, output_topic)
    """
    yield KAFKA_TOPICS


@pytest.fixture(scope="function")
def kafka_bootstrap_servers(kafka_server: typing.Tuple[subprocess.Popen, int]):
    """
    Used by tests that require both an input and an output topic
    """
    kafka_port = kafka_server[1]
    yield "localhost:{}".format(kafka_port)


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
def restore_sys_path():
    orig_vars = sys.path.copy()
    yield sys.path
    sys.path = orig_vars


@pytest.fixture(scope="function")
def import_mod(request: pytest.FixtureRequest, restore_sys_path):
    marker = request.node.get_closest_marker("import_mod")
    if marker is not None:
        mod_paths = marker.args[0]
        if not isinstance(mod_paths, list):
            mod_paths = [mod_paths]

        modules = []
        for mod_path in mod_paths:
            mod_dir, mod_fname = os.path.split(mod_path)
            mod_name, _ = os.path.splitext(mod_fname)

            sys.path.append(mod_dir)
            mod = importlib.import_module(mod_name)
            assert mod.__file__ == mod_path

            modules.append(mod)

        yield modules

    else:
        raise ValueError("import_mod fixture requires setting paths in markers: "
                         "`@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing/messages.py')])`")


def _reload_modules(modules: typing.List[typing.Any]):
    for mod in modules:
        importlib.reload(mod)


@pytest.fixture(scope="function")
def reload_modules(request: pytest.FixtureRequest):
    marker = request.node.get_closest_marker("reload_modules")
    modules = []
    if marker is not None:
        modules = marker.args[0]
        if not isinstance(modules, list):
            modules = [modules]

    _reload_modules(modules)
    yield
    _reload_modules(modules)


@pytest.fixture(scope="function")
def manual_seed():
    """
    Seeds the random number generators for the stdlib, PyTorch and NumPy.
    By default this will seed with a value of `42`, however this fixture also yields the seed function allowing tests to
    call this a second time, or seed with a different value.

    Use this fixture to ensure repeatability of a test that depends on randomness.
    Note: PyTorch only garuntees determanism on a per-GPU basis, resulting in some tests that might not be portable
    across GPU models.
    """
    from morpheus.utils import seed as seed_utils

    def seed_fn(seed=42):
        seed_utils.manual_seed(seed)

    seed_fn()
    yield seed_fn


@pytest.fixture(scope="function")
def chdir_tmpdir(request: pytest.FixtureRequest, tmp_path):
    """
    Executes a test in the tmp_path directory
    """
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


@pytest.fixture(scope="function")
def dataset(df_type: typing.Literal['cudf', 'pandas'], use_cpp: bool):
    """
    Yields a DatasetLoader instance with `df_type` as the default DataFrame type.
    """
    from utils import dataset_manager
    yield dataset_manager.DatasetManager(df_type=df_type)


@pytest.fixture(scope="function")
def dataset_pandas():
    """
    Yields a DatasetLoader instance with pandas as the default DataFrame type.
    """
    from utils import dataset_manager
    yield dataset_manager.DatasetManager(df_type='pandas')


@pytest.fixture(scope="function")
def dataset_cudf():
    """
    Yields a DatasetLoader instance with cudf as the default DataFrame type.
    """
    from utils import dataset_manager
    yield dataset_manager.DatasetManager(df_type='cudf')


@pytest.fixture(scope="function")
def filter_probs_df(dataset):
    """
    Shortcut fixture for loading the filter_probs.csv dataset
    """
    yield dataset["filter_probs.csv"]


def wait_for_camouflage(host="localhost", port=8000, timeout=5):

    start_time = time.time()
    cur_time = start_time
    end_time = start_time + timeout

    url = "http://{}:{}/ping".format(host, port)

    while cur_time - start_time <= timeout:
        timeout_epoch = min(cur_time + 1.0, end_time)

        try:
            request_timeout = max(timeout_epoch - cur_time, 0.1)
            r = requests.get(url, timeout=request_timeout)

            if (r.status_code == 200):
                if (r.json()['message'] == 'I am alive.'):
                    return True
                else:
                    warnings.warn(("Camoflage returned status 200 but had incorrect response JSON. "
                                   "Continuing to wait. Response JSON:\n%s"),
                                  r.json())

        except Exception:
            pass

        # Sleep up to the end time or max 1 second
        time.sleep(max(timeout_epoch - time.time(), 0.0))

        # Update current time
        cur_time = time.time()

    return False


def _set_pdeathsig(sig=signal.SIGTERM):
    """
    Helper function to ensure once parent process exits, its child processes will automatically die
    """

    def prctl_fn():
        libc = ctypes.CDLL("libc.so.6")
        return libc.prctl(1, sig)

    return prctl_fn


@pytest.fixture(scope="session")
def _camouflage_is_running():
    """
    Responsible for actually starting and shutting down Camouflage. This has the scope of 'session' so we only
    start/stop Camouflage once per testing session. Should not be used directly. Instead use `launch_mock_triton`

    Yields
    ------
    bool
        Whether or not we are using Camouflage or an actual Triton server
    """

    from utils import TEST_DIRS

    logger = logging.getLogger(f"morpheus.{__name__}")

    root_dir = TEST_DIRS.mock_triton_servers_dir
    startup_timeout = 5
    shutdown_timeout = 5

    launch_camouflage = os.environ.get('MORPHEUS_NO_LAUNCH_CAMOUFLAGE') is None
    is_running = False

    # First, check to see if camoflage is already open
    if (launch_camouflage):
        is_running = wait_for_camouflage(timeout=0.0)

        if (is_running):
            logger.warning("Camoflage already running. Skipping startup")
            launch_camouflage = False
            is_running = True

    # Actually launch camoflague
    if launch_camouflage:
        popen = None
        try:
            popen = subprocess.Popen(["camouflage", "--config", "config.yml"],
                                     cwd=root_dir,
                                     stderr=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL,
                                     preexec_fn=_set_pdeathsig(signal.SIGTERM))

            logger.info("Launched camouflage in %s with pid: %s", root_dir, popen.pid)

            if not wait_for_camouflage(timeout=startup_timeout):

                if popen.poll() is not None:
                    raise RuntimeError("camouflage server exited with status code={} details in: {}".format(
                        popen.poll(), os.path.join(root_dir, 'camouflage.log')))

                raise RuntimeError("Failed to launch camouflage server")

            # Must have been started by this point
            yield True

        except Exception:
            # Log the error and rethrow
            logger.exception("Error launching camouflage")
            raise
        finally:
            if popen is not None:
                logger.info("Killing camouflage with pid {}".format(popen.pid))

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

        yield is_running


@pytest.fixture(scope="function")
def launch_mock_triton(_camouflage_is_running):
    """
    Launches a mock triton server using camouflage (https://testinggospels.github.io/camouflage/) with a package
    rooted at `root_dir` and configured with `config`.

    This function will wait for up to `timeout` seconds for camoflauge to startup

    This function is a no-op if the `MORPHEUS_NO_LAUNCH_CAMOUFLAGE` environment variable is defined, which can
    be useful during test development to run camouflage by hand.
    """

    # Check if we are using Camouflage or not. If so, send the reset command to reset the state
    if _camouflage_is_running:
        # Reset the mock server (necessary to set counters = 0)
        r = requests.post("http://localhost:8000/reset", timeout=2.0)

        assert r.ok, "Failed to reset Camouflage server state"

    yield


@pytest.fixture(scope="session", autouse=True)
def configure_tests_logging(pytestconfig: pytest.Config):
    """
    Sets the base logging settings for the entire test suite to ensure logs are generated. Automatically detects if a
    debugger is attached and lowers the logging level to DEBUG.
    """
    import sys

    from morpheus.utils.logger import configure_logging

    log_level = logging.WARNING

    # Check if a debugger is attached. If so, choose DEBUG for the logging level. Otherwise, only WARN
    trace_func = sys.gettrace()

    if (trace_func is not None):
        trace_module = getattr(trace_func, "__module__", None)

        if (trace_module is not None and trace_module.find("pydevd") != -1):
            log_level = logging.DEBUG

    config_log_level = pytestconfig.getoption("log_level")

    # Overwrite the logging level if specified
    if (config_log_level is not None):
        log_level = logging.getLevelName(config_log_level)

    configure_logging(log_level=log_level)


def _wrap_set_log_level(log_level: int):
    from morpheus.utils.logger import set_log_level

    # Save the previous logging level
    old_level = set_log_level(log_level)

    yield

    set_log_level(old_level)


@pytest.fixture(scope="function")
def reset_loglevel():
    """
    Fixture restores the log level after running the given test.
    """
    import mrc

    from morpheus.utils.logger import set_log_level

    old_level = mrc.logging.get_level()

    yield

    set_log_level(old_level)


@pytest.fixture(scope="function")
def loglevel_debug():
    """
    Sets the logging level to `logging.DEBUG` for this function only.
    """
    _wrap_set_log_level(logging.DEBUG)


@pytest.fixture(scope="function")
def loglevel_info():
    """
    Sets the logging level to `logging.INFO` for this function only.
    """
    _wrap_set_log_level(logging.INFO)


@pytest.fixture(scope="function")
def loglevel_warn():
    """
    Sets the logging level to `logging.WARN` for this function only.
    """
    _wrap_set_log_level(logging.WARN)


@pytest.fixture(scope="function")
def loglevel_error():
    """
    Sets the logging level to `logging.ERROR` for this function only.
    """
    _wrap_set_log_level(logging.ERROR)


@pytest.fixture(scope="function")
def loglevel_fatal():
    """
    Sets the logging level to `logging.FATAL` for this function only.
    """
    _wrap_set_log_level(logging.FATAL)


# ==== DataFrame Fixtures ====
