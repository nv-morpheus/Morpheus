### Control Messages Submission Demo Setup

#### Introduction
This document will provide you instructions for setting up a Control Messages Submission GUI that enables users to create and submit control messages to a Kafka topic, which can then be used for training and evaluating a digital fingerprinting model (AutoEncoder). The Control Messages Submission GUI is a web-based application that provides a user-friendly interface for generating control messages, and it can be set up with the help of this guide. It provides step-by-step instructions for setting up the required dependencies for Kafka, Flask server, and endpoint URLs. By the end of this document, you will have a fully functional demo Control Messages Submission GUI that you can use for your digital fingerprinting workflow.

#### Requirements

To set up the Control Messages Submission GUI, you will need to install the following dependencies:

```
pip install flask confluent_kafka
```

#### Kafka Setup

To publish control messages to a Kafka topic, you will need to start the Kafka service first. Navigate to the `~/examples/digital_fingerprinting/`production directory and execute the following command:

```
docker-compose up kafka
```

##### Create Kafka Topic

Create a Kafka topic named `test_cm` to submit control messages. Run the following command to create the topic:
```
docker exec -it kafka kafka-topics --create --topic test_cm --bootstrap-server localhost:9092
```

To ensure that the topic is receiving messages, run the following command:
```
docker exec -it kafka kafka-console-consumer --topic test_cm --from-beginning --bootstrap-server localhost:9092
```

#### Flask Server Setup

To set up the Flask server for the Control Messages Submission GUI, navigate to the `bin` directory and execute the `start.sh` script:
```
bash start.sh
```

#### Access GUI
- `http://localhost:3000` : This URL provides flexibility to demonstrate the range of control message creation options.
    -   See for more information [here](./submit_messages.md)
- `http://localhost:3000/training` : This URL generates control messages exclusively for training purposes with some user-specified parameters.
    -   See [here](./training.md) for more details on the training GUI.
- `http://localhost:3000/review/results` : This URL is used to submit training messages after reviewing inference results.
    -   See for more information [here](review_results.md)
