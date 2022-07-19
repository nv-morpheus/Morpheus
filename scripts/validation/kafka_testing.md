## Pre-reqs
1. Launch Kafka using instructions from the [Quick Launch Kafka Cluster](../../CONTRIBUTING.md#quick-launch-kafka-cluster) section of [CONTRIBUTING.md](../../CONTRIBUTING.md)
1. Populate an environment variable `BROKER_LIST` with the IP:Ports of the nodes in the Kafka cluster. Ensure this environment variable is set in the terminals where Morpheus is executed:
    ```bash
    BROKER_LIST=$(HOST_IP=$KAFKA_ADVERTISED_HOST_NAME ./broker-list.sh)
    ```

## Simple Data Copying
1. Create a topic called "morpheus-copy-test" with only a single partition
    ```bash
    ./start-kafka-shell.sh $KAFKA_ADVERTISED_HOST_NAME
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-copy-test  --partitions 1 --bootstrap-server `broker-list.sh`
    ```
1. Open a new terminal and launch a pipeline to listen to Kafka, from the root of the Morpheus repo run:
    ```bash
    morpheus --log_level=DEBUG run pipeline-nlp from-kafka --input_topic morpheus-copy-test --bootstrap_servers "${BROKER_LIST}" deserialize monitor --description read serialize to-file --filename=/tmp/morpheus-copy-test.csv --overwrite
    ```
1. Open a new terminal and launch a Kafka writer process:
    ```bash
    morpheus --log_level=DEBUG run pipeline-nlp from-file --filename=tests/tests_data/filter_probs.csv deserialize  serialize --exclude='^_ts_' to-kafka --output_topic morpheus-copy-test --bootstrap_servers "${BROKER_LIST}"
    ```
    The `tests/tests_data/filter_probs.csv` contains 20 lines of data and the pipeline should complete rather quickly (less than 5 seconds).

1. Return to the first terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Cntrl-C.
1. If successful our output file `/tmp/morpheus-copy-test.csv` should be identicle to `tests/tests_data/filter_probs.csv` with the addition of cuDF's ID column. To verify the output we will strip the new ID column and pipe the output to `diff`:
    ```bash
    scripts/validation/strip_first_csv_col.py /tmp/morpheus-copy-test.csv | diff -q --ignore-all-space tests/tests_data/filter_probs.csv -
    ```

## Partitioned Data Copying
Same as above, but we cannot depend on the ordering of the records being preserved.
1. Create a topic called "morpheus-copy-test-p" with three partitions
    ```bash
    ./start-kafka-shell.sh $KAFKA_ADVERTISED_HOST_NAME
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-copy-test-p  --partitions 3 --bootstrap-server `broker-list.sh`
    ```
1. Open a new terminal and launch a pipeline to listen to Kafka, from the root of the Morpheus repo run:
    ```bash
    morpheus --log_level=DEBUG run pipeline-nlp from-kafka --input_topic morpheus-copy-test-p --bootstrap_servers "${BROKER_LIST}" deserialize monitor --description read serialize to-file --filename=/tmp/morpheus-copy-test-p.csv --overwrite
    ```
1. Open a new terminal and launch a Kafka writer process:
    ```bash
    morpheus --log_level=DEBUG run pipeline-nlp from-file --filename=tests/tests_data/filter_probs.csv deserialize  serialize --exclude='^_ts_' to-kafka --output_topic morpheus-copy-test-p --bootstrap_servers "${BROKER_LIST}"
    ```
    The `tests/tests_data/filter_probs.csv` contains 20 lines of data and the pipeline should complete rather quickly (less than 5 seconds).

1. Return to the first terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Cntrl-C.
1. If successful our output file `/tmp/morpheus-copy-test-p.csv` should contain the same records as those in `tests/tests_data/filter_probs.csv` however they are most likely out of order. To verify the output we will strip the new ID column and compare the sorted outputs:
    ```bash
    scripts/validation/strip_first_csv_col.py /tmp/morpheus-copy-test-p.csv | sort | diff -q --ignore-all-space - <(sort tests/tests_data/filter_probs.csv)
    ```

## Validation Pipeline
For this test we are going to split the ABP validation pipeline into three pipelines roughly this will look like:

Pipe 1:

    from-file -> to-kafka topic=morpheus-abp-pre


Pipe 2:

    from-kafka topic=morpheus-abp-pre -> <actual inference pipeline> -> to-kafka topic=morpheus-abp-post

Pipe 3:

    from-kafka topic=morpheus-abp-post -> to-file

1. Create two Kafka topics both with only a single partition.
    ```bash
    ./start-kafka-shell.sh $KAFKA_ADVERTISED_HOST_NAME
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-abp-pre  --partitions 1 --bootstrap-server `broker-list.sh`
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-abp-post  --partitions 1 --bootstrap-server `broker-list.sh`
    ```
1. In a new terminal launch Triton:
    ```bash
    docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models \
        nvcr.io/nvidia/tritonserver:22.02-py3 \
        tritonserver --model-repository=/models/triton-model-repo \
                     --exit-on-error=false \
                     --model-control-mode=explicit \
                     --load-model abp-nvsmi-xgb
    ```
1. Open a new terminal and launch a pipeline to listen to Kafka, from the root of the Morpheus repo run:
    ```bash
    morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 \
        pipeline-fil \
        from-kafka --input_topic morpheus-abp-post --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Read" \
        deserialize \
        serialize \
        to-file --filename=/tmp/val-abp-kafka.json --overwrite
    ```
1. Open a new terminal and launch the inference pipeline which will both listen and write to kafka:
    ```bash
    morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 \
        pipeline-fil \
        from-kafka --input_topic morpheus-abp-pre --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Read" \
        deserialize \
        preprocess \
        inf-triton --model_name=abp-nvsmi-xgb --server_url="localhost:8000" --force_convert_inputs=True \
        monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
        add-class \
        serialize \
        to-kafka --output_topic morpheus-abp-post --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Write"
    ```
1. Open a new terminal and launch the pipeline which will write the source data into Kafka:
    ```bash
    morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 \
        pipeline-fil \
        from-file --filename=${MORPHEUS_ROOT}/models/datasets/validation-data/abp-validation-data.jsonlines \
        deserialize \
        serialize \
        to-kafka --output_topic morpheus-abp-pre --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Write"
    ```
    This command should execute quickly writing `1242` records and should complete in less than 5 seconds.
1. Shutdown the other two pipelines once their respective monitor stages have recorded `1242` records.
1. Output file should be identicle to the `models/datasets/validation-data/abp-validation-data.jsonlines` file:
    ```bash
    diff -q --ignore-all-space ${MORPHEUS_ROOT}/models/datasets/validation-data/abp-validation-data.jsonlines /tmp/val-abp-kafka.json
    ```
