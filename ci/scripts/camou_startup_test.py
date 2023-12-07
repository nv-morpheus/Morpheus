import ctypes
import os
import signal
import subprocess
import time
import typing
import warnings

import requests

ROOT_DIR = "/home/dagardner/work/morpheus/tests/mock_rest_server"


def _set_pdeathsig(sig=signal.SIGTERM):
    """
    Helper function to ensure once parent process exits, its child processes will automatically die
    """

    def prctl_fn():
        libc = ctypes.CDLL("libc.so.6")
        return libc.prctl(1, sig)

    return prctl_fn


def wait_for_camouflage(host="localhost", port=8000, timeout=10):

    start_time = time.time()
    cur_time = start_time
    end_time = start_time + timeout

    url = f"http://{host}:{port}/ping"

    while cur_time - start_time <= timeout:
        timeout_epoch = min(cur_time + 2.0, end_time)

        try:
            request_timeout = max(timeout_epoch - cur_time, 1)
            print(f"issueing request with timeout={request_timeout}")
            resp = requests.get(url, timeout=request_timeout)

            if (resp.status_code == 200):
                if (resp.json()['message'] == 'I am alive.'):
                    return True

                warnings.warn(("Camoflage returned status 200 but had incorrect response JSON. Continuing to wait. "
                               "Response JSON:\n%s"),
                              resp.json())

        except Exception:
            pass

        # Sleep up to the end time or max 1 second
        sleep_time = max(timeout_epoch - time.time(), 0.0)
        print(f"sleep_time={sleep_time}")
        time.sleep(sleep_time)

        # Update current time
        cur_time = time.time()

    print(f"Runtime = {cur_time-start_time}")
    return False


def _stop_camouflage(popen: subprocess.Popen, shutdown_timeout: int = 5):

    print(f"Killing camouflage with pid {popen.pid}")

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


def _start_camouflage(root_dir: str,
                      host: str = "localhost",
                      port: int = 8000) -> typing.Tuple[bool, typing.Optional[subprocess.Popen]]:
    startup_timeout = 10

    launch_camouflage = os.environ.get('MORPHEUS_NO_LAUNCH_CAMOUFLAGE') is None
    is_running = False

    # First, check to see if camoflage is already open
    if (launch_camouflage):
        is_running = wait_for_camouflage(host=host, port=port, timeout=0.0)

        if (is_running):
            warnings.warn("Camoflage already running. Skipping startup")
            launch_camouflage = False
            is_running = True

    # Actually launch camoflague
    if launch_camouflage:
        popen = None
        try:
            # pylint: disable=subprocess-popen-preexec-fn,consider-using-with
            popen = subprocess.Popen(["camouflage", "--config", "config.yml"],
                                     cwd=root_dir,
                                     stderr=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL,
                                     preexec_fn=_set_pdeathsig(signal.SIGTERM))
            # pylint: enable=subprocess-popen-preexec-fn,consider-using-with

            print(f"Launched camouflage in {root_dir} with pid: {popen.pid}")

            def read_logs():
                camouflage_log = os.path.join(root_dir, 'camouflage.log')
                if os.path.exists(camouflage_log):
                    with open(camouflage_log, 'r', encoding='utf-8') as f:
                        return f.read()
                return ""

            if not wait_for_camouflage(host=host, port=port, timeout=startup_timeout):

                if popen.poll() is not None:
                    raise RuntimeError(f"camouflage server exited with status code={popen.poll()}\n{read_logs()}")

                raise RuntimeError(f"Failed to launch camouflage server\n{read_logs()}")

            # Must have been started by this point
            return (True, popen)

        except Exception:
            # Log the error and rethrow
            print("Error launching camouflage")
            if popen is not None:
                _stop_camouflage(popen)
            raise

    else:

        return (is_running, None)


def main():
    itr = 0
    is_running = True
    log_file = os.path.join(ROOT_DIR, 'camouflage.log')
    while is_running:
        if os.path.exists(log_file):
            os.remove(log_file)

        print(f"\n**********\nStarting iteration {itr}")
        (is_running, popen) = _start_camouflage(root_dir=ROOT_DIR, port=8080)

        if not is_running:
            print(f"Camouflage failed to start on iteration {itr}")

        if popen is not None:
            _stop_camouflage(popen)

        itr += 1


if __name__ == "__main__":
    main()
