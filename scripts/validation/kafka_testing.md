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
