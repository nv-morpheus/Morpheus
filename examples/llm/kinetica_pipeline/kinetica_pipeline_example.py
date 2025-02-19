import json
import logging
import os
import random

import cudf

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.modules import to_control_message  # noqa: F401 # pylint: disable=unused-import
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE
from morpheus_llm.service.vdb.kinetica_vector_db_service import KineticaVectorDBService
from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage

logger = logging.getLogger(__name__)


def get_test_df(num_input_rows):
    df = cudf.DataFrame({
        "id": list(range(num_input_rows)),
        "embeddings": [[random.random() for _ in range(3)] for _ in range(num_input_rows)],
        "metadata": [json.dumps({"metadata": f"Sample metadata for row {i}"}) for i in range(num_input_rows)],
    })

    return df


def main():
    host = os.getenv("kinetica_host", "http://localhost:9191")
    username = os.getenv("username", "")
    password = os.getenv("password", "")
    schema = os.getenv("schema", "")

    config = Config()
    config.execution_mode = ExecutionMode.GPU

    kinetica_db_service = KineticaVectorDBService(host, user=username, password=password, kinetica_schema=schema)
    collection_name = "test_collection"
    collection_name = f"{schema}.{collection_name}" if schema is not None and len(
        schema) > 0 else f"ki_home.{collection_name}"

    columns = [
        ["id", "long", "primary_key"],
        ["embeddings", "bytes", "vector(3)"],
        ["metadata", "string", "json"],
    ]
    kinetica_db_service.create(collection_name, type=columns)

    df = get_test_df(10)
    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE, "module_name": "to_control_message", "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    # Step 1: Create a pipeline
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [df]))
    pipeline.add_stage(
        LinearModulesStage(config,
                           to_cm_module_config,
                           input_port_name="input",
                           output_port_name="output",
                           output_type=ControlMessage))

    pipeline.add_stage(
        WriteToVectorDBStage(
            config,
            kinetica_db_service,
            "test_collection"
        )
    )

    pipeline.run()


if __name__ == "__main__":
    main()
