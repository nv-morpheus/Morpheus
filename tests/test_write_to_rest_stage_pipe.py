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

from functools import partial
from io import StringIO
from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.io.serializers import df_to_stream_json
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.write_to_rest_stage import WriteToRestStage
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.type_aliases import DataFrameType
from utils import make_mock_response
from utils.dataset_manager import DatasetManager


def _df_to_buffer(df: DataFrameType, lines: bool) -> StringIO:
    buffer = StringIO()
    df_to_stream_json(df=df, stream=buffer, lines=lines)
    buffer.seek(0)
    return buffer


def _df_to_url(lines: bool, base_url: str, endpoint: str, df: DataFrameType) -> dict:
    return {'url': f"{base_url}{endpoint}/{df.index[0]}", 'data': _df_to_buffer(df=df, lines=lines)}


@pytest.mark.slow
@pytest.mark.use_cudf
@pytest.mark.parametrize("method", [HTTPMethod.POST, HTTPMethod.PUT])
@pytest.mark.parametrize("lines", [False, True])
@pytest.mark.parametrize("max_rows_per_payload", [10000, 10])
@pytest.mark.parametrize("use_df_to_url", [False, True])
@mock.patch("requests.Session")
@mock.patch("time.sleep")
def test_write_to_rest_stage_pipe(mock_sleep: mock.MagicMock,
                                  mock_request_session: mock.MagicMock,
                                  config: Config,
                                  dataset: DatasetManager,
                                  method: HTTPMethod,
                                  lines: bool,
                                  max_rows_per_payload: int,
                                  use_df_to_url: bool):
    make_mock_response(mock_request_session)
    if lines:
        expected_content_type = MimeTypes.TEXT.value
    else:
        expected_content_type = MimeTypes.JSON.value

    if use_df_to_url:
        df_to_request_kwargs_fn = partial(_df_to_url, lines)
    else:
        df_to_request_kwargs_fn = None

    df = dataset['filter_probs.csv']

    if max_rows_per_payload > len(df):
        num_expected_requests = 1
    else:
        num_expected_requests = len(df) // max_rows_per_payload

    expected_payloads = []
    rows_serialized = 0
    while rows_serialized < len(df):
        expected_payload = _df_to_buffer(df=df[rows_serialized:rows_serialized + max_rows_per_payload], lines=lines)
        expected_payloads.append(expected_payload)
        rows_serialized += max_rows_per_payload

    assert len(expected_payloads) == num_expected_requests

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        WriteToRestStage(config,
                         base_url="http://fake.nvidia.com",
                         endpoint="/data",
                         method=method,
                         request_timeout_secs=42,
                         query_params={'unit': 'test'},
                         lines=lines,
                         max_rows_per_payload=max_rows_per_payload,
                         df_to_request_kwargs_fn=df_to_request_kwargs_fn))
    pipe.run()

    assert mock_request_session.request.call_count == num_expected_requests

    mocked_calls = mock_request_session.request.call_args_list
    for i, call in enumerate(mocked_calls):
        # The `data` argument is a StringIO buffer which prevents us from testing directly for equality
        called_buffer = call.kwargs['data']
        assert called_buffer.read() == expected_payloads[i].read()
        called_buffer.seek(0)

        expected_url = 'http://fake.nvidia.com/data'
        if df_to_request_kwargs_fn is not None:
            expected_url = f"{expected_url}/{i*max_rows_per_payload}"

        assert call == mock.call(method=method.value,
                                 headers={"Content-Type": expected_content_type},
                                 timeout=42,
                                 params={'unit': 'test'},
                                 url=expected_url,
                                 data=called_buffer)

    mock_sleep.assert_not_called()
