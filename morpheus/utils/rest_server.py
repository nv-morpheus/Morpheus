# Copyright (c) 2023, NVIDIA CORPORATION.
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

import logging
import multiprocessing as mp
import queue
import typing

import gunicorn.app.base
from flask import Flask
from flask import request
from flask.typing import ResponseReturnValue
from flask.views import View

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    'bind': '127.0.0.1:8080',
    'keepalive': 0,
    'loglevel': 'WARNING',
    'preload_app': True,
    'proc_name': 'morpheus_rest_server',
    'reuse_port': True,
    'timeout': 0,
    'worker_class': 'sync',
    'workers': 1
}


class MorpheusRestView(View):
    init_every_request = False

    def __init__(self, logger: logging.Logger, queue: mp.Queue, success_status=201, queue_timeout=30):
        self._logger = logger
        self._queue = queue
        self._queue_timeout = queue_timeout
        self._success_status = success_status

    def dispatch_request(self) -> ResponseReturnValue:
        """
        Receives a request and puts it in the queue.
        Note this does not perform any validation on the request pyaload, it is possible that incoming data
        which would later fail to be processed by Morpheus is accepted by this endpoint.
        """

        # This should work the same for both POST & PUT
        if request.is_json and request.content_length is not None and request.content_length > 0:
            try:
                self._logger.debug("Received request with content length %s", request.content_length)
                data = request.data.decode("utf-8")
                self._queue.put(data, block=True, timeout=self._queue_timeout)
                return "", self._success_status
            except queue.Full:
                return "Request queue is full", 503
            except Exception as e:
                return f"Error processing request: {e}", 500
        else:
            return "Request is not JSON", 400


class MorpheusRestServer(gunicorn.app.base.BaseApplication):

    def __init__(self, app: typing.Callable, options: dict):
        logger.debug("Starting gunicorn server with options: %s", options)
        self.application = app
        self.options = options
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def _start_rest_server(options: dict, queue: mp.Queue):
    app = Flask('morpheus_rest_server')

    # TODO make this configurable
    app.logger.setLevel(logging.DEBUG)
    app.add_url_rule('/submit',
                     methods=["POST", "PUT"],
                     view_func=MorpheusRestView.as_view('request_handler', logger=app.logger, queue=queue))
    server = MorpheusRestServer(app, options)
    server.run()


# TODO: Currently `options` configures the gunicorn server, we need a second config to configure Flask specifically the
# endpoint(s) and supported methods.
def start_rest_server(options: dict = None) -> typing.Tuple[mp.Process, mp.Queue]:
    """
    Starts a REST server.
    """

    # gunicorn is opinionated and doesn't really support being embedded in another app, so we need to start it in a
    # separate process.
    # TODO: Although gunicorn is a good choice we should look into cherrypy and other options.
    server_options = DEFAULT_OPTIONS.copy()
    server_options.update(options or {})
    queue = mp.Queue()

    process = mp.Process(target=_start_rest_server,
                         name='morpheus_rest_server',
                         kwargs={
                             'options': server_options, 'queue': queue
                         })
    return (process, queue)
