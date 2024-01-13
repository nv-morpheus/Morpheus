import os
import types

import cudf
import pytest
from _utils import TEST_DIRS
from _utils import assert_results

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.slow
@pytest.mark.use_python
@pytest.mark.use_cudf
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/common/web_scraper_module.py'))
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
