from morpheus.pipeline.pipeline import LinearPipeline
from morpheus.config import Config
from morpheus.stages.general.source_stage import SourceStage
from morpheus.stages.general.filter_stage import FilterStage
from morpheus.stages.general.serialize_stage import SerializeStage
from python.morpheus_llm.morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.utils.logging import configure_logging
from morpheus.utils.producer import IterableProducer
from morpheus.utils.type_support import numpy_to_cudf

# Import Milvus services from Morpheus
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBResourceService, MilvusVectorDBService

import numpy as np
import json

def main():
    # Step 1: Configure logging
    configure_logging(log_level="INFO")

    # Step 2: Initialize Morpheus Config
    config = Config()

    # Step 3: Setup Milvus services
    milvus_resource_service = MilvusVectorDBResourceService(host="127.0.0.1", port="19530")
    collection_name = "morpheus_vectors"
    vector_dim = 3  # Example: 3-dimensional vector embeddings

    # Step 4: Initialize the Milvus VectorDBService
    milvus_service = MilvusVectorDBService(
        collection_name=collection_name,
        dim=vector_dim,
        resource_service=milvus_resource_service,
    )

    # Step 5: Create a pipeline
    pipeline = LinearPipeline(config)

    # Step 6: Define source stage
    def data_generator():
        for i in range(5):
            embedding = np.random.random(vector_dim).tolist()
            metadata = {"id": i, "label": f"example_{i}"}
            yield {"embedding": embedding, "metadata": metadata}

    pipeline.add_stage(
        SourceStage(
            config=config,
            source=IterableProducer(data_generator())
        )
    )

    # Step 7: Add filter stage
    pipeline.add_stage(
        FilterStage(
            config=config,
            filter_func=lambda msg: msg["embedding"] is not None  # Only process messages with valid embeddings
        )
    )

    # Step 8: Serialize stage
    pipeline.add_stage(
        SerializeStage(
            config=config,
            to_dict_func=lambda msg: {
                "embedding": msg["embedding"],
                "metadata": msg["metadata"]
            }
        )
    )

    # Step 9: Add WriteToVectorDBStage for Milvus
    pipeline.add_stage(
        WriteToVectorDBStage(
            config=config,
            vdb_service=milvus_service,
            embedding_field="embedding",
            metadata_field="metadata"
        )
    )

    # Step 10: Execute the pipeline
    pipeline.run()

if __name__ == "__main__":
    main()
