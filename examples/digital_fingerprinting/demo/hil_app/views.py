import json
import logging

from flask import jsonify
from flask import render_template
from flask import request
from hil_app.kafka_helper import publish_message

from . import app

logger = logging.getLogger(__name__)


@app.route('/', methods=["GET", "POST"])
def submit_messages():
    if request.method == "POST":
        control_message = request.form.get("control_message")
        logger.error(control_message)
        publish_message(control_message)
        data = {
            "Data": "Successfully published task to kafka topic.",
        }
        return jsonify(data)

    if request.method == "GET":
        return render_template("home.html")
