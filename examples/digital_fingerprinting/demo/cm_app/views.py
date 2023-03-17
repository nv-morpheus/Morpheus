import logging

from cm_app.helper import KafkaWriter
from cm_app.helper import generate_success_message
from cm_app.helper import process_cm
from confluent_kafka import Producer
from flask import render_template
from flask import request

from . import app

kafka_writer = None


@app.before_first_request
def setup():
    app.logger.setLevel(logging.INFO)
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    app.logger.info("Initialized Kafka producer")
    global kafka_writer
    kafka_writer = KafkaWriter(kafka_topic="test_cm", batch_size=1, producer=producer)
    app.logger.info("Initialized Kafka writer")


@app.route('/', methods=["GET", "POST"])
def submit_messages():

    if request.method == "POST":
        control_messages_json = process_cm(request)
        global kafka_writer
        kafka_writer.write_data(control_messages_json)
        sucess_message = generate_success_message(control_messages_json)
        return sucess_message

    if request.method == "GET":
        return render_template("submit_messages.html")


@app.route('/training', methods=["GET", "POST"])
def training():

    if request.method == "POST":
        control_messages_json = process_cm(request)
        global kafka_writer
        kafka_writer.write_data(control_messages_json)
        sucess_message = generate_success_message(control_messages_json)
        return sucess_message

    if request.method == "GET":
        return render_template("training.html")


@app.route('/review/results', methods=["GET"])
def reviewresults():

    if request.method == "GET":
        return render_template("review/results.html")
