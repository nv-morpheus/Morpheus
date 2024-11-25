import csv
import logging
import random

from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.config import Config
from morpheus.stages.postprocess.serialize_stage import SerializeStage

from morpheus.stages.input.in_memory_data_generation_stage import InMemoryDataGenStage

from morpheus.stages.preprocess.drop_null_stage import DropNullStage
from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
# from morpheus.utils.logging import configure_logging
# from morpheus.utils.type_support import numpy_to_cudf

# Import Milvus services from Morpheus
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBResourceService, MilvusVectorDBService

import numpy as np
import json

from morpheus.utils.logger import configure_logging

from morpheus.stages.input.file_source_stage import FileSourceStage

from morpheus.stages.output.write_to_file_stage import WriteToFileStage

from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

from morpheus.config import ExecutionMode

logger = logging.getLogger(__name__)

def generate_csv(file_name, num_rows):
    """
    Generates a CSV file with fields:
    - PK: Primary Key (integer)
    - metadata: Sample text metadata (string)
    - vector: A random vector of 3 dimensions (array of floats)

    :param file_name: Name of the output CSV file
    :param num_rows: Number of rows to generate
    """
    with open(file_name, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'metadata', 'embedding']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        for pk in range(1, num_rows + 1):
            # Generate random metadata
            metadata = f"Sample metadata for row {pk}"

            # Generate a random vector of 3 dimensions
            vector = [round(random.uniform(-1.0, 1.0), 4) for _ in range(3)]

            # Write the row
            writer.writerow({
                'id': pk,
                'metadata': metadata,
                'embedding': vector
            })


def main(input_file_name: str):
    # Step 1: Configure logging

    # Step 2: Initialize Morpheus Config
    config = Config()
    config.execution_mode = ExecutionMode.GPU

    # Step 3: Setup Milvus services
    milvus_db_service = MilvusVectorDBService("http://127.0.0.1:19530")
    milvus_resource_service = milvus_db_service.load_resource("test_collection")
    collection_name = "test_collection"
    vector_dim = 3  # Example: 3-dimensional vector embeddings

    # Step 5: Create a pipeline
    pipeline = LinearPipeline(config)

    # Step 6: Define source stage
    def data_generator():
        for i in range(5):
            embedding = np.random.random(vector_dim).tolist()
            metadata = {"id": i, "label": f"example_{i}"}
            yield {"embedding": embedding, "metadata": metadata}

    pipeline.set_source(FileSourceStage(config, filename=input_file_name))

    pipeline.add_stage(
        DropNullStage(config, "embedding")
    )

    # pipeline.add_stage(WriteToFileStage(config, filename="output_file.csv", overwrite=True))

    pipeline.add_stage(DeserializeStage(config))
    # Step 9: Add WriteToVectorDBStage for Milvus
    pipeline.add_stage(
        WriteToVectorDBStage(
            config,
            milvus_db_service,
            "test_collection"
        )
    )

    pipeline.build()

    # Step 10: Execute the pipeline
    pipeline.run()

if __name__ == "__main__":
    file_name = "test.csv"
    generate_csv(file_name, 1000)
    main(file_name)
