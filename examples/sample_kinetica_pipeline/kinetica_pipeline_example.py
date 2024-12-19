import csv
import logging
import random
import cudf

from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.config import Config
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.type_utils import exec_mode_to_df_type_str

from morpheus.stages.input.in_memory_data_generation_stage import InMemoryDataGenStage

from morpheus.stages.preprocess.drop_null_stage import DropNullStage
from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage

# from morpheus.utils.logging import configure_logging
# from morpheus.utils.type_support import numpy_to_cudf

# Import Milvus services from Morpheus
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBResourceService, MilvusVectorDBService
from morpheus_llm.service.vdb.kinetica_vector_db_service import KineticaVectorDBResourceService, KineticaVectorDBService

import numpy as np
import json

from morpheus.utils.logger import configure_logging

from morpheus.stages.input.file_source_stage import FileSourceStage

from morpheus.stages.output.write_to_file_stage import WriteToFileStage

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
        "age": [random.randint(20, 40) for i in range(num_input_rows)],
        "embedding": [[random.random() for _ in range(3)] for _ in range(num_input_rows)]
    })

    return df


def main(input_file_name: str):
    # Step 1: Configure logging

    # Step 2: Initialize Morpheus Config
    config = Config()
    config.execution_mode = ExecutionMode.GPU

    # milvus_db_service = MilvusVectorDBService("https://in03-c87c25d216da0ac.serverless.gcp-us-west1.cloud.zilliz.com", user="db_c87c25d216da0ac", password="Cv3;^~HaY.>~>!)H", token="1c80242758bbfc207773c9a731421d9d96e269ac3ef41d87b40725f53795e1305489827dd310f0e55fb886ba0ea15898244de182")
    kinetica_db_service = KineticaVectorDBService("https://demo72.kinetica.com/_gpudb", user="amukherjee", password="Kinetica1!")
    # milvus_resource_service = milvus_db_service.load_resource("test_collection")
    kinetica_resource_service = kinetica_db_service.load_resource("test_collection")
    collection_name = "test_collection"
    vector_dim = 3  # Example: 3-dimensional vector embeddings

    source_df = read_file_to_df(input_file_name, df_type=exec_mode_to_df_type_str(config.execution_mode))
    print(source_df.shape[0])

    # Step 1: Create a pipeline
    pipeline = LinearPipeline(config)

    # # Step 6: Define source stage
    # def data_generator():
    #     for i in range(5):
    #         embedding = np.random.random(vector_dim).tolist()
    #         metadata = {"id": i, "label": f"example_{i}"}
    #         yield {"embedding": embedding, "metadata": metadata}

    pipeline.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipeline.add_stage(DeserializeStage(config))

    pipeline.add_stage(
        WriteToVectorDBStage(
            config,
            kinetica_db_service,
            "test_collection"
        )
    )

    pipeline.build()

    pipeline.run()

if __name__ == "__main__":
    file_name = "test.csv"
    generate_csv(file_name)
    main(file_name)
