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

import asyncio
import queue
import typing
from http import HTTPStatus
from io import StringIO

import pytest

from _utils import assert_results
from _utils import make_url
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.io.serializers import df_to_stream_json
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.http_server_source_stage import HttpServerSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.http_utils import request_with_retry


async def make_request(pipe: LinearPipeline,
                       response_queue: queue.Queue,
                       method: HTTPMethod,
                       accept_status: HTTPStatus,
                       url: str,
                       payload: typing.Any,
                       content_type: str):
    attempt = 0
    while not pipe._is_started and attempt < 2:
        await asyncio.sleep(1)
        attempt += 1

    if not pipe._is_started:
        raise RuntimeError("HttpServerSourceStage did not start")

    # Not strictly needed, but we don't have a good way of knowing when the server is ready to accept requests
    # Adding this sleep here just lowers the likely-hood of seeing a logged warning on the first failed request.
    await asyncio.sleep(0.1)

    (_, response) = request_with_retry(
        {
            'method': method.value,
            'url': url,
            'data': payload,
            'timeout': 5.0,
            'allow_redirects': False,
            'headers': {
                "Content-Type": content_type
            }
        },
        accept_status_codes=[accept_status],
        respect_retry_after_header=False)
    response_queue.put_nowait(response)


async def run_pipe_and_request(pipe: LinearPipeline,
                               response_queue: queue.Queue,
                               method: HTTPMethod,
                               accept_status: HTTPStatus,
                               url: str,
                               payload: typing.Any,
                               content_type: str):
    await asyncio.gather(pipe.run_async(),
                         make_request(pipe, response_queue, method, accept_status, url, payload, content_type))


@pytest.mark.slow
@pytest.mark.parametrize("lines", [False, True])
def test_http_server_source_stage_pipe(config: Config, dataset_cudf: DatasetManager, lines: bool):
    endpoint = '/test'
    port = 8088
    method = HTTPMethod.POST
    accept_status = HTTPStatus.OK
    url = make_url(port, endpoint)

    if lines:
        content_type = MimeTypes.TEXT.value
    else:
        content_type = MimeTypes.JSON.value

    df = dataset_cudf['filter_probs.csv']
    num_records = len(df)
    buf = df_to_stream_json(df, StringIO(), lines=lines)
    buf.seek(0)

    payload = buf.read()

    pipe = LinearPipeline(config)
    pipe.set_source(
        HttpServerSourceStage(config=config,
                              port=port,
                              endpoint=endpoint,
                              method=method,
                              accept_status=accept_status,
                              lines=lines,
                              stop_after=num_records))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, df))

    response_queue = queue.SimpleQueue()

    asyncio.run(run_pipe_and_request(pipe, response_queue, method, accept_status, url, payload, content_type))
    assert_results(comp_stage.get_results())

    response = response_queue.get_nowait()

    assert response.status_code == accept_status.value
    assert response.headers["Content-Type"] == MimeTypes.TEXT.value
    assert response.text == ""
