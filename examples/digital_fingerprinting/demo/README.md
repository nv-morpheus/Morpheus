
#### Kafka Setup

Start Kafka service.

```
cd ~/examples/digital_fingerprinting/production

docker-compose up kafka zookeeper
```

##### Create Kafka Topic
```
docker exec -it kafka kafka-topics --create --topic test_cm --bootstrap-server localhost:9092
```

Verify created topic is receiving messages. 
```
docker exec -it kafka kafka-console-consumer --topic test_cm --from-beginning --bootstrap-server localhost:9092
```

#### Flask Server Setup

```
pip install flask
```

```
cd ~/examples/digital_fingerprinting/demo/bin

bash start.sh
```

#### Endpoint URL's
```
http://localhost:3000
http://localhost:3000/training
```

