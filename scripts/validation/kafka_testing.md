This document walks through manual testing of the Kafka functionality in Morpheus. There are also several automated tests which are run as part of the CI process. To run the tests locally we will need to install a few dependencies needed for the tests:
```bash
mamba install -c conda-forge "openjdk=11.0.15"
npm install -g camouflage-server@0.9
mkdir -p ${MORPHEUS_ROOT}/.cache
git clone https://gitlab.com/karolinepauls/pytest-kafka.git ${MORPHEUS_ROOT}/.cache/pytest-kafka
cd ${MORPHEUS_ROOT}/.cache/pytest-kafka
python setup.py develop
cd ${MORPHEUS_ROOT}
```

Then run the Kafka tests with:
```bash
pytest --run_slow --run_kafka
```

## Pre-reqs
1. Create the `${MORPHEUS_ROOT}/.tmp` dir (this dir is already listed in the `.gitignore` file).
    ```bash
    mkdir -p ${MORPHEUS_ROOT}/.tmp
    ```
1. Launch Kafka using instructions from the [Quick Launch Kafka Cluster](../../CONTRIBUTING.md#quick-launch-kafka-cluster) section of [CONTRIBUTING.md](../../CONTRIBUTING.md) following steps 1-6.

1. The testing steps below will require four separate terminal windows. Each will need to have the `KAFKA_ADVERTISED_HOST_NAME`, `BROKER_LIST` and `MORPHEUS_ROOT` environment variables set. In the example below both morpheus and kafka-docker repositories have been checked out into the `~work` directory, replacing these paths with the location of your checkouts.
    ```bash
    export MORPHEUS_ROOT=~/work/morpheus
    export KAFKA_ADVERTISED_HOST_NAME=$(docker network inspect bridge | jq -r '.[0].IPAM.Config[0].Gateway')
    export BROKER_LIST=$(HOST_IP=$KAFKA_ADVERTISED_HOST_NAME ~/work/kafka-docker/broker-list.sh)
    ```
1. Open two new terminals and start the Kafka docker container in each:
    ```bash
    docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
         -e HOST_IP=$KAFKA_ADVERTISED_HOST_NAME -e ZK=$2 \
         -v ${MORPHEUS_ROOT}:/workspace wurstmeister/kafka /bin/bash
    ```

    Leave these terminals open the testing steps will refer to these as the "first Kafka terminal" and "second Kafka terminal", all commands executed from these terminals will be within the kafka container.

1. Open two new terminals and navigate to the root of the Morpheus repo. The first terminal will be referred to as the "Morpheus terminal" and will be used for running Morpheus pipelines and verifying output. The second terminal will be referred to as the "Triton terminal" used for launching Triton.

### File descriptors
If you receive errors from Kafka such as `Too many open files`, you may need to increase the maximum number of open file descriptors. To check the current file descriptor limit run:
```bash
ulimit -n
```

To increase the limit (in this example to `4096`):
```bash
ulimit -n 4096
```



## Simple Data Copying
### Checking KafkaSourceStage
#### Single Partition Topic Test
1. From the first Kafka terminal, create a topic called "morpheus-src-copy-test" with only a single partition.
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-src-copy-test  --partitions 1 --bootstrap-server `broker-list.sh`
    ```

1. From the Morpheus terminal launch a pipeline to listen to Kafka:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-kafka --input_topic morpheus-src-copy-test --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Read" \
        deserialize \
        serialize \
        to-file --include-index-col=false --filename=${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test.csv --overwrite
    ```

1. Return to the first Kafka terminal and run:
    ```bash
    cat /workspace/tests/tests_data/filter_probs.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-src-copy-test --broker-list=`broker-list.sh` -
    ```

1. Return to the Morpheus terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Ctrl-C.

1. If successful the output file `.tmp/morpheus-src-copy-test.csv` should be identical to `tests/tests_data/filter_probs.csv`. Verify:
    ```bash
    diff -q --ignore-all-space ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv ${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test.csv
    ```

#### Partitioned Topic Test
1. From the first Kafka terminal create a new topic named `morpheus-src-copy-test-p` with three partitions:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-src-copy-test-p --partitions 3 --bootstrap-server `broker-list.sh`
    ```

1. From the Morpheus terminal run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-kafka --input_topic morpheus-src-copy-test-p --bootstrap_servers "${BROKER_LIST}" \
        deserialize \
        monitor --description "Kafka Read" \
        serialize \
        to-file --include-index-col=false --filename=${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test-p.csv --overwrite
    ```

1. Return to the first Kafka terminal and run:
    ```bash
    cat /workspace/tests/tests_data/filter_probs.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-src-copy-test-p --broker-list=`broker-list.sh` -
    ```

1. Return to the Morpheus terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Ctrl-C.

1. If successful the output file `.tmp/morpheus-src-copy-test-p.csv` should contain the same records as those in `tests/tests_data/filter_probs.csv` however they are most likely out of order. To verify the output we will compare the sorted outputs:
    ```bash
    diff -q --ignore-all-space <(sort tests/tests_data/filter_probs.csv) <(sort .tmp/morpheus-src-copy-test-p.csv)
    ```


### Checking WriteToKafkaStage
#### Single Partition Topic Test
1. From the first Kafka terminal create a topic called "morpheus-sink-copy-test" with only a single partition, and start a consumer on that topic:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-sink-copy-test  --partitions 1 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-sink-copy-test \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/morpheus-sink-copy-test.jsonlines
    ```

1. From the Morpheus terminal run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-file --filename=${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv \
        deserialize  \
        serialize \
        to-kafka --output_topic morpheus-sink-copy-test --bootstrap_servers "${BROKER_LIST}"
    ```
    The `tests/tests_data/filter_probs.csv` contains 20 lines of data and the pipeline should complete rather quickly (less than 5 seconds).

1. The Kafka consumer we started in step #1 won't give us any sort of indication as to how many records have been consumed, we will indirectly check the progress by counting the rows in the output file. Once the Morpheus pipeline completes check the number of lines in the output:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test.jsonlines
    ```

1. Once all 20 lines have been written to the output file, verify the contents with:
    ```bash
    diff -q --ignore-all-space <(cat ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test.jsonlines | jq --sort-keys) <(cat ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.jsonlines | jq --sort-keys)
    ```
    Note the usage of `jq --sort-keys` which will reformat the json output, sorting the keys, this ensures that `{"a": 5, "b": 6}` and `{"b": 6,   "a": 5}` are considered equivalent.

1. Stop the consumer in the first Kafka terminal.

#### Partitioned Topic Test
1. From the first Kafka terminal create a new topic named "morpheus-sink-copy-test-p" with three partitions, and start a consumer on that topic:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-sink-copy-test-p --partitions 3 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-sink-copy-test-p \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/morpheus-sink-copy-test-p.jsonlines
    ```

1. From the Morpheus terminal run:
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
    Note due to the multiple partitions the consumer most likely received records out of order, so we are comparing the sorted output of both files.

1. Stop the consumer in the first Kafka terminal.


## ABP Validation Pipeline
For this test we are going to replace the from & to file stages from the ABP validation pipeline with Kafka stages, reading input data from a Kafka topic named "morpheus-abp-pre" and writing results to a topic named "morpheus-abp-post"

1. From the first Kafka terminal create two topics both with only a single partition, and launch a consumer listening to the `morpheus-abp-post` topic.
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-abp-pre  --partitions 1 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-abp-post  --partitions 1 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-abp-post \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/val_kafka_abp-nvsmi-xgb.jsonlines
    ```

1. From the Triton terminal run:
    ```bash
    docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v ${MORPHEUS_ROOT}/models:/models \
        nvcr.io/nvidia/tritonserver:22.08-py3 \
        tritonserver --model-repository=/models/triton-model-repo \
                     --exit-on-error=false \
                     --model-control-mode=explicit \
                     --load-model abp-nvsmi-xgb
    ```

1. From the Morpheus terminal launch the inference pipeline which will both listen and write to kafka:
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

1. From the second Kafka terminal launch a producer to feed the `morpheus-abp-pre` topic with the input data:
    ```bash
    cat /workspace/models/datasets/validation-data/abp-validation-data.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-abp-pre --broker-list=`broker-list.sh` -
    ```
    This command should execute quickly writing `1242` records and should complete in less than 5 seconds.

1. Return to the Morpheus terminal. Once the `Kafka Write` monitor reports that `1242` messages have been written, shutdown Morpheus with Ctrl-C. We can check the number of lines in the output file:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/val_kafka_abp-nvsmi-xgb.jsonlines
    ```

1. Once all `1242` lines have been written to the output file, verify the contents with:
    ```bash
    diff -q --ignore-all-space <(cat ${MORPHEUS_ROOT}/models/datasets/validation-data/abp-validation-data.jsonlines | jq --sort-keys) <(cat ${MORPHEUS_ROOT}/.tmp/val_kafka_abp-nvsmi-xgb.jsonlines | jq --sort-keys)
    ```

1. Return to the first Kafka terminal and stop the consumer.
1. Return to the Triton Terminal and stop Triton.

## DFP (Hammah) Validation Pipeline
### User123
For this test we are going to replace the `to-file` stage from the Hammah validation pipeline with the `to-kafka` stage using a topic named "morpheus-hammah-user123". Note: this pipeline requires a custom `UserMessageMeta` class which the `from-kafka` stage is currently unable to generate, for that reason the `CloudTrailSourceStage` remains in-place.

1. From the first Kafka terminal create the `morpheus-hammah-user123` topic, and launch a consumer listening to it:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-hammah-user123 --partitions 1 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-hammah-user123 \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/val_kafka_hammah-user123-pytorch.jsonlines
    ```

1. From the Morpheus terminal launch the pipeline which will write results to kafka:
    ```bash
    morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=false \
      pipeline-ae --userid_filter="user123" --userid_column_name="userIdentitysessionContextsessionIssueruserName" \
      from-cloudtrail --input_glob="${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-*.csv" \
      train-ae --train_data_glob="${MORPHEUS_ROOT}/models/datasets/training-data/hammah-*.csv"  --seed 42 \
      preprocess \
      inf-pytorch \
      add-scores \
      timeseries --resolution=1m --zscore_threshold=8.0 --hot_start \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      serialize --exclude='event_dt|tlsDetailsclientProvidedHostHeader' \
      to-kafka --output_topic morpheus-hammah-user123 --bootstrap_servers "${BROKER_LIST}" \
      monitor --description "Kafka Write"
    ```

    This pipeline should complete in approximately 10 seconds, with the Kafka monitor stage recording `847` messages written to Kafka.

1. The Kafka consumer we started in step #1 won't give us any sort of indication as to how many records have been consumed, we will indirectly check the progress by counting the rows in the output file. Once the Morpheus pipeline completes check the number of lines in the output:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/val_kafka_hammah-user123-pytorch.jsonlines
    ```

1. Once all `847` rows have been written, return to the first Kafka terminal and stop the consumer with Ctrl-C.

1. Verify the output with, expect to see `38` unmatched rows:
    ```bash
    ${MORPHEUS_ROOT}/scripts/compare_data_files.py \
        ${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-user123-validation-data.csv \
        ${MORPHEUS_ROOT}/.tmp/val_kafka_hammah-user123-pytorch.jsonlines \
        --index_col="_index_" --exclude "event_dt" --rel_tol=0.1
    ```

### Role-g
Similar to the Hammah User123 test, we are going to replace the `to-file` stage from the Hammah validation pipeline with the `to-kafka` stage using a topic named "morpheus-hammah-role-g".

1. From the first Kafka terminal create the `morpheus-hammah-role-g` topic, and launch a consumer listening to it:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-hammah-role-g --partitions 1 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-hammah-role-g \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/val_kafka_hammah-role-g-pytorch.jsonlines
    ```

1. From the Morpheus terminal launch the pipeline which will write results to kafka:
    ```bash
    morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=false \
      pipeline-ae --userid_filter="role-g" --userid_column_name="userIdentitysessionContextsessionIssueruserName" \
      from-cloudtrail --input_glob="${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-*.csv" \
      train-ae --train_data_glob="${MORPHEUS_ROOT}/models/datasets/training-data/hammah-*.csv"  --seed 42 \
      preprocess \
      inf-pytorch \
      add-scores \
      timeseries --resolution=10m --zscore_threshold=8.0 \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      serialize --exclude='event_dt|tlsDetailsclientProvidedHostHeader' \
      to-kafka --output_topic morpheus-hammah-role-g --bootstrap_servers "${BROKER_LIST}" \
      monitor --description "Kafka Write"
    ```

    This pipeline should complete in approximately 10 seconds, with the Kafka monitor stage recording `314` messages written to Kafka.

1. The Kafka consumer we started in step #1 won't give us any sort of indication as to how many records have been consumed, we will indirectly check the progress by counting the rows in the output file. Once the Morpheus pipeline completes check the number of lines in the output:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/val_kafka_hammah-role-g-pytorch.jsonlines
    ```

1. Once all `314` rows have been written, return to the first Kafka terminal and stop the consumer with Ctrl-C.

1. Verify the output with, all rows should match:
    ```bash
    ${MORPHEUS_ROOT}/scripts/compare_data_files.py \
        ${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-role-g-validation-data.csv \
        ${MORPHEUS_ROOT}/.tmp/val_kafka_hammah-role-g-pytorch.jsonlines  \
        --index_col="_index_" --exclude "event_dt" --rel_tol=0.15
    ```

## Phishing Validation Pipeline
For this test we are going to replace the from & to file stages from the Phishing validation pipeline with Kafka stages, reading input data from a Kafka topic named "morpheus-phishing-pre" and writing results to a topic named "morpheus-phishing-post"

1.  From the first Kafka terminal create the two topics both with only a single partition, and launch a consumer listening to the `morpheus-phishing-post` topic.
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-phishing-pre --partitions 1 --bootstrap-server `broker-list.sh`
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-phishing-post --partitions 1 --bootstrap-server `broker-list.sh`
    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-phishing-post \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/val_kafka_phishing.jsonlines
    ```

1. From the Triton terminal launch Triton with:
    ```bash
    docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v ${MORPHEUS_ROOT}/models:/models \
        nvcr.io/nvidia/tritonserver:22.08-py3 \
        tritonserver --model-repository=/models/triton-model-repo \
                     --exit-on-error=false \
                     --model-control-mode=explicit \
                     --load-model phishing-bert-onnx
    ```

1. From the Morpheus terminal launch the inference pipeline which will both listen and write to kafka:
    ```bash
    morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 \
        pipeline-nlp --model_seq_length=128 --labels_file=${MORPHEUS_ROOT}/morpheus/data/labels_phishing.txt \
        from-kafka --input_topic morpheus-phishing-pre --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Read" \
        deserialize \
        preprocess --vocab_hash_file=${MORPHEUS_ROOT}/morpheus/data/bert-base-uncased-hash.txt \
            --truncation=True --do_lower_case=True --add_special_tokens=False \
        inf-triton --model_name=phishing-bert-onnx --server_url="localhost:8000" --force_convert_inputs=True \
        monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
        add-class --label=pred --threshold=0.7 \
        serialize \
        to-kafka --output_topic morpheus-phishing-post --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Write"
    ```

1. From the second Kafka terminal launch a Kafka producer to feed the `morpheus-phishing-pre` topic with the input data:
    ```bash
    cat /workspace/models/datasets/validation-data/phishing-email-validation-data.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-phishing-pre --broker-list=`broker-list.sh` -
    ```
    This command should execute quickly writing `1010` records and should complete in less than 5 seconds.

1. Return to the Morpheus terminal. The pipeline will take anywhere from 2 to 5 minutes to complete. Once the `Kafka Write` monitor has reported that `1010` messages have been written, shutdown Morpheus with Ctrl-C. We can check the number of lines in the output file:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/val_kafka_phishing.jsonlines
    ```

1. Once all `1010` rows have been written, return to the first Kafka terminal and stop the consumer with Ctrl-C.

1. Verify the output with, expect to see `43` un-matched rows:
    ```bash
    ${MORPHEUS_ROOT}/scripts/compare_data_files.py \
        ${MORPHEUS_ROOT}/models/datasets/validation-data/phishing-email-validation-data.jsonlines \
        ${MORPHEUS_ROOT}/.tmp/val_kafka_phishing.jsonlines
    ```

1. Stop Triton

## Sid Validation Pipeline
For this test we are going to replace the file stage from the Sid validation pipeline with the to-kafka stage writing results to a topic named "morpheus-sid-post".
Note: Due to the complexity of the input data and a limitation of the cudf reader we will need to keep the from-file source reading data as CSV.

1. From the Kafka terminal create a topic named `morpheus-sid-post` and launch a consumer listening to the topic.
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-sid-post --partitions 1 --bootstrap-server `broker-list.sh`
    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-sid-post \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/val_kafka_sid.jsonlines
    ```

1. From the Triton terminal launch Triton:
    ```bash
    docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v ${MORPHEUS_ROOT}/models:/models \
        nvcr.io/nvidia/tritonserver:22.08-py3 \
        tritonserver --model-repository=/models/triton-model-repo \
                     --exit-on-error=false \
                     --model-control-mode=explicit \
                     --load-model sid-minibert-onnx
    ```

1. From the Morpheus terminal launch the inference pipeline which will both listen and write to kafka:
    ```bash
    morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 \
        pipeline-nlp --model_seq_length=256 \
        from-file --filename=${MORPHEUS_ROOT}/models/datasets/validation-data/sid-validation-data.csv \
        deserialize \
        preprocess --vocab_hash_file=${MORPHEUS_ROOT}/morpheus/data/bert-base-uncased-hash.txt \
            --truncation=True --do_lower_case=True --add_special_tokens=False \
        inf-triton --model_name=sid-minibert-onnx --server_url="localhost:8000" --force_convert_inputs=True \
        monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
        add-class --prefix="si_" \
        serialize --exclude "id" --exclude "^_ts_" \
        to-kafka --output_topic morpheus-sid-post --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Write"
    ```

1. The pipeline will take approximately 2 minutes to complete. We can check the number of lines in the output file:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/val_kafka_sid.jsonlines
    ```

1. Once all `2000` rows have been written, return to the first Kafka terminal and stop the consumer with Ctrl-C.

1. Verify the output with, expect to see `25` un-matched rows:
    ```bash
    ${MORPHEUS_ROOT}/scripts/compare_data_files.py \
        ${MORPHEUS_ROOT}/models/datasets/validation-data/sid-validation-data.csv \
        ${MORPHEUS_ROOT}/.tmp/val_kafka_sid.jsonlines
    ```

1. Stop Triton

## Optional Cleanup
### Delete all topics
1. Return to the first Kafka terminal and within the container run:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --list --bootstrap-server `broker-list.sh` | xargs -I'{}' $KAFKA_HOME/bin/kafka-topics.sh --delete --bootstrap-server `broker-list.sh` --topic='{}'
    ```

### Shutdown Kafka
1. Exit from both Kafka terminals.
1. From the root of the `kafka-docker` repo run (in the host OS not inside a container):
    ```bash
    docker-compose stop
    docker-compose rm
    ```
