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

import queue
import threading
import time
import typing
from http import HTTPStatus
from io import StringIO

import pytest
import requests

from _utils import make_url
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.io.serializers import df_to_stream_json
from morpheus.messages import MessageMeta
from morpheus.stages.input.http_server_source_stage import HttpServerSourceStage
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import MimeTypes


class GetNext(threading.Thread):

    def __init__(self, msg_queue: queue.Queue, generator: typing.Iterator[MessageMeta]):
        threading.Thread.__init__(self)
        self._generator = generator
        self._msg_queue = msg_queue
        self._exception = None

    def run(self):
        try:
            msg = next(self._generator)
            self._msg_queue.put_nowait(msg)
        except Exception as e:
            print(f"Exception in GetNext thread: {e}")
            self._exception = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self._exception:
            raise self._exception


@pytest.mark.slow
@pytest.mark.use_python
@pytest.mark.parametrize("lines", [False, True])
def test_generate_frames(config: Config, dataset_pandas: DatasetManager, lines: bool):
    # The _generate_frames() method is only used when C++ mode is disabled
    endpoint = '/test'
    port = 8088
    method = HTTPMethod.POST
    accept_status = HTTPStatus.OK
    url = make_url(port, endpoint)

    if lines:
        content_type = MimeTypes.TEXT.value
    else:
        content_type = MimeTypes.JSON.value

    df = dataset_pandas['filter_probs.csv']
    buf = df_to_stream_json(df, StringIO(), lines=lines)
    buf.seek(0)

    payload = buf.read()

    stage = HttpServerSourceStage(config=config,
                                  port=port,
                                  endpoint=endpoint,
                                  method=method,
                                  accept_status=accept_status,
                                  lines=lines)

    generate_frames = stage._generate_frames()
    msg_queue = queue.SimpleQueue()

    get_next_thread = GetNext(msg_queue, generate_frames)
    get_next_thread.start()

    attempt = 0
    while not stage._processing and get_next_thread.is_alive() and attempt < 2:
        time.sleep(0.1)
        attempt += 1

    assert stage._processing
    assert get_next_thread.is_alive()

    response = requests.request(method=method.value,
                                url=url,
                                data=payload,
                                timeout=5.0,
                                allow_redirects=False,
                                headers={"Content-Type": content_type})

    result_msg = msg_queue.get(timeout=5.0)
    get_next_thread.join()

    assert response.status_code == accept_status.value
    assert response.headers["Content-Type"] == MimeTypes.TEXT.value
    assert response.text == ""

    dataset_pandas.assert_compare_df(df, result_msg.df)


@pytest.mark.parametrize("invalid_method", [HTTPMethod.GET, HTTPMethod.PATCH])
def test_constructor_invalid_method(config: Config, invalid_method: HTTPMethod):
    with pytest.raises(ValueError):
        HttpServerSourceStage(config=config, method=invalid_method)


@pytest.mark.parametrize("invalid_accept_status", [HTTPStatus.CONTINUE, HTTPStatus.FOUND])
def test_constructor_invalid_accept_status(config: Config, invalid_accept_status: HTTPStatus):
    with pytest.raises(ValueError):
        HttpServerSourceStage(config=config, accept_status=invalid_accept_status)
