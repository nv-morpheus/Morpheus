import json
import logging

logger = logging.getLogger(__name__)


class KafkaWriter:

    def __init__(self, kafka_topic, batch_size, producer):
        self._kafka_topic = kafka_topic
        self._batch_size = batch_size
        self._producer = producer

    @property
    def producer(self):
        return self._producer

    def write_data(self, message):
        self.producer.produce(self._kafka_topic, message.encode('utf-8'))
        if len(self.producer) >= self._batch_size:
            logger.info(
                "Batch reached, calling poll... producer unsent: %s",
                len(self.producer),
            )
            self.producer.flush()

    def close(self):
        logger.info("Closing kafka writer...")
        if self.producer is not None:
            self.producer.flush()
        logger.info("Closing kafka writer...Done")


def process_cm(request):
    control_messages_json = request.form.get("control-messages-json")

    logging.info("Received control message: {}".format(control_messages_json))

    return control_messages_json


def generate_success_message(control_messages_json):
    sucess_message = {
        "status": "Successfully published control message to kafka topic.",
        "status_code": 200,
        "control_messages": json.loads(control_messages_json)
    }

    sucess_message = json.dumps(sucess_message, indent=4)
    return sucess_message
