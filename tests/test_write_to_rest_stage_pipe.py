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
from utils import make_mock_response
from utils.dataset_manager import DatasetManager


@pytest.mark.slow
@pytest.mark.use_cudf
@pytest.mark.parametrize("method", [HTTPMethod.POST, HTTPMethod.PUT])
@pytest.mark.parametrize("lines", [False, True])
@mock.patch("requests.Session")
@mock.patch("time.sleep")
def test_write_to_rest_stage_pipe(mock_sleep: mock.MagicMock,
                                  mock_request_session: mock.MagicMock,
                                  config: Config,
                                  dataset: DatasetManager,
                                  method: HTTPMethod,
                                  lines: bool):
    make_mock_response(mock_request_session)
    if lines:
        expected_content_type = MimeTypes.TEXT.value
    else:
        expected_content_type = MimeTypes.JSON.value

    df = dataset['filter_probs.csv']

    expected_payload = StringIO()
    df_to_stream_json(df=df, stream=expected_payload, lines=lines)
    expected_payload.seek(0)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        WriteToRestStage(config,
                         base_url="http://fake.nvidia.com",
                         endpoint="/data",
                         method=method,
                         request_timeout_secs=42,
                         query_params={'unit': 'test'},
                         lines=lines))
    pipe.run()

    # The `data` argument is a StringIO buffer which prevents us from testing directly for equality
    mock_request_session.request.assert_called_once()

    called_buffer = mock_request_session.request.call_args.kwargs['data']
    assert called_buffer.read() == expected_payload.read()
    called_buffer.seek(0)
    mock_request_session.request.assert_called_once_with(method=method.value,
                                                         headers={"Content-Type": expected_content_type},
                                                         timeout=42,
                                                         params={'unit': 'test'},
                                                         url='http://fake.nvidia.com/data',
                                                         data=called_buffer)

    mock_sleep.assert_not_called()
