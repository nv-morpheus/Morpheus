## Pre-reqs
1. Launch Kafka using instructions from the [Quick Launch Kafka Cluster](../../CONTRIBUTING.md#quick-launch-kafka-cluster) section of [CONTRIBUTING.md](../../CONTRIBUTING.md)
1. Populate an environment variable `BROKER_LIST` with the IP:Ports of the nodes in the Kafka cluster. Ensure this environment variable is set in the terminals where Morpheus is executed:
    ```bash
    BROKER_LIST=$(HOST_IP=$KAFKA_ADVERTISED_HOST_NAME ./broker-list.sh)
    ```

## Simple Data Copying
### Checking KafkaSourceStage
#### Single Partition Topic Test
1. Open a new terminal and create a topic called "morpheus-src-copy-test" with only a single partition
    ```bash
    docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
         -e HOST_IP=$KAFKA_ADVERTISED_HOST_NAME -e ZK=$2 \
         -v ${MORPHEUS_ROOT}:/workspace wurstmeister/kafka /bin/bash
    
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-src-copy-test  --partitions 1 --bootstrap-server `broker-list.sh`
    ```
    Keep this shell & container open you will need it in later steps.

1. Open a new terminal and launch a pipeline to listen to Kafka, from the root of the Morpheus repo run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-kafka --input_topic morpheus-src-copy-test --bootstrap_servers "${BROKER_LIST}" \
        deserialize \
        monitor --description "Kafka Read" \
        serialize \
        to-file --include-index-col=false --filename=${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test.csv --overwrite
    ```

1. Return to the Kafka terminal and run:
    ```bash
    cat /workspace/tests/tests_data/filter_probs.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-src-copy-test --broker-list=`broker-list.sh` -
    ```

1. Return to the Morpheus terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Cntrl-C.

1. If successful the output file `.tmp/morpheus-src-copy-test.csv` should be identicle to `tests/tests_data/filter_probs.csv`. Verify:
    ```bash
    diff -q --ignore-all-space ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv ${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test.csv
    ```

#### Partitioned Topic Test
1. From the Kafka terminal create a new topic named "morpheus-src-copy-test-p" with three partitions:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-src-copy-test-p --partitions 3 --bootstrap-server `broker-list.sh`
    ```

1. Open a new terminal and launch a pipeline to listen to Kafka, from the root of the Morpheus repo run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-kafka --input_topic morpheus-src-copy-test-p --bootstrap_servers "${BROKER_LIST}" \
        deserialize \
        monitor --description "Kafka Read" \
        serialize \
        to-file --include-index-col=false --filename=${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test-p.csv --overwrite
    ```

1. Return to the Kafka terminal and run:
    ```bash
    cat /workspace/tests/tests_data/filter_probs.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-src-copy-test-p --broker-list=`broker-list.sh` -
    ```

1. Return to the Morpheus terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Cntrl-C.

1. If successful the output file `.tmp/morpheus-src-copy-test-p.csv` should contain the same records as those in `tests/tests_data/filter_probs.csv` however they are most likely out of order. To verify the output we will compare the sorted outputs:
    ```bash
    diff -q --ignore-all-space <(sort tests/tests_data/filter_probs.csv) <(sort .tmp/morpheus-src-copy-test-p.csv)
    ```


### Checking WriteToKafkaStage
#### Single Partition Topic Test
1. Open a new terminal and create a topic called "morpheus-sink-copy-test" with only a single partition, and start a consumer on that topic:
    ```bash
    docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
         -e HOST_IP=$KAFKA_ADVERTISED_HOST_NAME -e ZK=$2 \
         -v ${MORPHEUS_ROOT}:/workspace wurstmeister/kafka /bin/bash
    
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-sink-copy-test  --partitions 1 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-sink-copy-test \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/morpheus-sink-copy-test.jsonlines
    ```

1. Open a new terminal and from the Morpheus root run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-file --filename=${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv \
        deserialize  \
        serialize \
        to-kafka --output_topic morpheus-sink-copy-test --bootstrap_servers "${BROKER_LIST}"
    ```
    The `tests/tests_data/filter_probs.csv` contains 20 lines of data and the pipeline should complete rather quickly (less than 5 seconds).

1. The Kafka consumer we started in step #1 won't give us any sort of indication that it has concluded we will indirectly check the progress by counting the rows in the output file. Once the Morpheus pipeline completes check the number of lines in the output:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test.jsonlines
    ```

1. Once all 20 lines have been written to the output file, verify the contents with:
    ```bash
    diff -q --ignore-all-space <(cat ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test.jsonlines | jq --sort-keys) <(cat ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.jsonlines | jq --sort-keys)
    ```
    Note the usage of `jq --sort-keys` which will reformat the json outut, sorting the keys, this ensures that `{"a": 5, "b": 6}` and `{"b": 6,   "a": 5}` are considered equivelant.

1. Stop the consumer in the Kafka terminal.

#### Partitioned Topic Test
1. From the Kafka terminal create a new topic named "morpheus-sink-copy-test-p" with three partitions, and start a consumer on that topic:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-sink-copy-test-p --partitions 3 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-sink-copy-test-p \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/morpheus-sink-copy-test-p.jsonlines
    ```

1. Open a new terminal and from the Morpheus root run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-file --filename=${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv \
        deserialize  \
        serialize \
        to-kafka --output_topic morpheus-sink-copy-test-p --bootstrap_servers "${BROKER_LIST}"
    ```
    The `tests/tests_data/filter_probs.csv` contains 20 lines of data and the pipeline should complete rather quickly (less than 5 seconds).

1. The Kafka consumer we started in step #1 won't give us any sort of indication that it has concluded we will indirectly check the progress by counting the rows in the output file. Once the Morpheus pipeline completes check the number of lines in the output:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test-p.jsonlines
    ```

1. Once all 20 lines have been written to the output file, verify the contents with:
    ```bash
    diff -q --ignore-all-space <(sort ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test-p.jsonlines | jq --sort-keys) <(sort ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.jsonlines | jq --sort-keys)
    ```
    Note due to the multiple partitions the consumer most likely receieved records out of order, so we are comparing the sorted output of both files.

1. Stop the consumer in the Kafka terminal.


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
