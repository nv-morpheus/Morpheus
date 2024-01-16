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


import os
import types
from unittest.mock import patch, MagicMock

import cudf
import fsspec
import pytest
from _utils import TEST_DIRS
from _utils import assert_results

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage

# Mock dependencies
file_meta_mock = MagicMock()
text_converter_mock = MagicMock()

# TODO

@pytest.mark.use_python
@pytest.mark.use_cudf
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/common/content_extractor_module.py'))
def test_http_client_source_stage_pipe(config: Config, mock_rest_server: str, import_mod: types.ModuleType):
    url = f"{mock_rest_server}/www/index"

    df = cudf.DataFrame({"link": [url]})
    df_expected = cudf.DataFrame({"link": [url], "page_content": "website title some paragraph"})

    web_scraper_definition = import_mod.WebScraperInterface.get_definition("web_scraper",
                                                                           module_config={"web_scraper_config": {
                                                                               "link_column": "link", "chunk_size": 100,
                                                                               "enable_cache": False,
                                                                               "cache_path": "./.cache/http/RSSDownloadStage.sqlite",
                                                                               "cache_dir": "./.cache/llm/rss"}})

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(LinearModulesStage(config,
                                      web_scraper_definition,
                                      input_type=MessageMeta,
                                      output_type=MessageMeta,
                                      input_port_name="input",
                                      output_port_name="output"))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, compare_df=df_expected))
    pipe.run()

    print(comp_stage.get_messages())

    assert_results(comp_stage.get_results())


# 1. Test with Mocked Files and Converters
def test_parse_files_with_mocked_files():
    with patch('your_module.get_file_meta', return_value=file_meta_mock), \
            patch('your_module.TextConverter', return_value=text_converter_mock):
        open_files = [MagicMock(spec=fsspec.core.OpenFile) for _ in range(5)]
        expected_data = [{'content': 'mock content'}] * len(open_files)
        text_converter_mock.convert.return_value = expected_data

        result = your_module.parse_files(open_files)

        assert isinstance(result, MessageMeta)
        assert len(result.df) == len(open_files)
        assert result.df.to_dict('records') == expected_data


# 2. Test Handling of Exceptions During File Processing
def test_parse_files_with_exception():
    with patch('your_module.get_file_meta', side_effect=Exception("Error")), \
            patch('your_module.logger') as logger_mock:
        open_files = [MagicMock(spec=fsspec.core.OpenFile) for _ in range(2)]

        result = your_module.parse_files(open_files)

        assert logger_mock.error.called
        assert isinstance(result, MessageMeta)
        assert result.df.empty


# 3. Test Batch Processing
def test_parse_files_batch_processing():
    batch_size = 2
    open_files = [MagicMock(spec=fsspec.core.OpenFile) for _ in range(5)]

    # Modify your_module.batch_size accordingly
    your_module.batch_size = batch_size

    result = your_module.parse_files(open_files)

    assert len(result.df) == len(open_files)  # Assuming each file results in one row


# 4. Test Processing Different File Types
@pytest.mark.parametrize("file_type, converter", [("pdf", pdf_converter_mock), ("txt", text_converter_mock)])
def test_parse_files_different_file_types(file_type, converter):
    with patch('your_module.get_file_meta', return_value={"file_type": file_type}), \
            patch(f'your_module.{converter.__class__.__name__}', return_value=converter):
        open_files = [MagicMock(spec=fsspec.core.OpenFile) for _ in range(2)]
        converter.convert.return_value = [{'content': 'mock content'}]

        result = your_module.parse_files(open_files)

        assert converter.convert.called
        assert len(result.df) == len(open_files)
