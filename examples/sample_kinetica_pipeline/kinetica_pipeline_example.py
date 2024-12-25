import csv
import logging
import random
import cudf
import os

from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.type_utils import exec_mode_to_df_type_str
from morpheus.modules import to_control_message  # noqa: F401 # pylint: disable=unused-import
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE

from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage

from morpheus.messages import ControlMessage

# from morpheus.utils.logging import configure_logging
# from morpheus.utils.type_support import numpy_to_cudf

# Import Milvus services from Morpheus
from morpheus_llm.service.vdb.kinetica_vector_db_service import KineticaVectorDBService

import json

from morpheus.utils.logger import configure_logging

from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

from morpheus.config import ExecutionMode

logger = logging.getLogger(__name__)

def generate_random_vector(dim=3):
    """Generate a random vector of specified dimensions."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def generate_csv(file_path, num_records=10):
    """
    Generate a CSV file with the specified format.

    Parameters:
        file_path (str): Path to the output CSV file.
        num_records (int): Number of records to generate.
    """
    records = []

    for i in range(1, num_records + 1):
        vector = generate_random_vector()
        metadata = json.dumps({"metadata": f"Sample metadata for row {i}"})
        records.append([i, str(vector), metadata])

    # Write records to CSV file
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["id", "embeddings", "metadata"])
        # Write data
        writer.writerows(records)

    print(f"CSV file '{file_path}' with {num_records} records generated successfully.")


def get_test_df(num_input_rows):
    df = cudf.DataFrame({
        "id": list(range(num_input_rows)),
        "embeddings": [[random.random() for _ in range(3)] for _ in range(num_input_rows)],
        "metadata": [json.dumps({"metadata": f"Sample metadata for row {i}"}) for i in range(num_input_rows)],
    })

    return df


def main(input_file_name: str):
    host = os.getenv("kinetica_host", "http://localhost:9191")
    username = os.getenv("username", "")
    password = os.getenv("password", "")
    schema = os.getenv("schema", "")
    # Step 1: Configure logging

    # Step 2: Initialize Morpheus Config
    config = Config()
    config.execution_mode = ExecutionMode.GPU

    kinetica_db_service = KineticaVectorDBService(host, user=username, password=password, kinetica_schema=schema)
    collection_name = "test_collection"
    collection_name = f"{schema}.{collection_name}" if schema is not None and len(
        schema) > 0 else f"ki_home.{collection_name}"

    vector_dim = 3  # Example: 3-dimensional vector embeddings

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
    file_name = "test.csv"
    generate_csv(file_name)
    main(file_name)
