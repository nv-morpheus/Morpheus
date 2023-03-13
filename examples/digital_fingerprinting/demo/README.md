
#### Kafka Setup

```
git clone https://github.com/conduktor/kafka-stack-docker-compose.git
```

```
cd kafka-stack-docker-compose

docker-compose -f zk-single-kafka-single.yml up

docker exec -it kafka1 bash

kafka-topics --create --topic test_cm --bootstrap-server localhost:9092

kafka-console-consumer --topic test_cm --from-beginning --bootstrap-server localhost:9092
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
http://localhost:3000/training
http://localhost:3000
```

