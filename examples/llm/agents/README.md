# Morpheus LLM Agents Example

<Insert Overview>

## Background on Agents

<Insert Background on LLM Agents and how they work>

## Simple Example

<Insert Simple Example description>

### Running the Simple Example

```bash
cd ${MORPHEUS_ROOT}/llm

python main.py agents simple
```

## Kafka Example

<Insert Kafka Example description>

### Running the Kafka Example

First, a Kafka cluster must be created in order to run the Kafka example. The cluster is required to allow the persistent pipeline to accept queries for the LLM agents. The Kafka cluster can be created using the following command:

<Insert instructions on starting a kafka cluster>

Once the Kafka cluster is running, the Kafka example can be run using the following command:

```bash
cd ${MORPHEUS_ROOT}/llm

python main.py agents kafka
```

After the pipeline is running, we need to send messages to the pipeline using the Kafka message bus. In a separate terminal, run the following command:

```bash
# Set the bootstrap server variable
export BOOTSTRAP_SERVER=$(broker-list.sh)

# Create the input and output topics
kafka-topics.sh --bootstrap-server ${BOOTSTRAP_SERVER} --create --topic input

# Update the partitions
kafka-topics.sh --bootstrap-server ${BOOTSTRAP_SERVER} --alter --topic input --partitions 3
```

Now, we can send messages to the pipeline using the following command:

```bash
kafka-console-producer.sh --bootstrap-server ${BOOTSTRAP_SERVER} --topic input
```

This will open up a prompt allowing any JSON to be pasted into the terminal. The JSON should be formatted as follows:

```json
{"question": "<Your question here>"}
```

For example:
```json
{"question": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"}
{"question": "What is the height of the tallest mountain in feet divided by 2.23? Do not round your answer"}
{"question": "Who is the current leader of Japan? What is the largest prime number that is smaller that their age? Just say the number."}
```
