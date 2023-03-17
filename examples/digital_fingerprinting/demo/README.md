### GUI Setup for Submitting Control Messages

#### Kafka Setup

Start Kafka service to publish control messages to kafka topic

```
cd ~/examples/digital_fingerprinting/production

docker-compose up kafka zookeeper
```

##### Create Kafka Topic

Create Kafka topic `test_cm` to submit control messages from `cm_app`.
```
docker exec -it kafka kafka-topics --create --topic test_cm --bootstrap-server localhost:9092
```

Make sure the topic you created is getting messages.
```
docker exec -it kafka kafka-console-consumer --topic test_cm --from-beginning --bootstrap-server localhost:9092
```

#### Flask Server Setup

Install flask python module to run the demo.

```
pip install flask
```

Navigate to the bin directory and execute start script.
```
cd ~/examples/digital_fingerprinting/demo/bin

bash start.sh
```

#### Endpoint URL's
Flexibility to demonstrate the range of control message creation options.
```
http://localhost:3000
```
Generates control messages for training purposes exclusively with some user-specified parameters.
```
http://localhost:3000/training
```

Submit training messages after reviewing inference results
```
http://localhost:3000/review/results
```
