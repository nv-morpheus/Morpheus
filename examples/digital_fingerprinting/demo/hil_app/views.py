import json
import logging

from flask import jsonify
from flask import render_template
from flask import request
from hil_app.kafka_helper import publish_message

from . import app

logging.basicConfig()
logger = logging.getLogger("logger")


@app.route('/', methods=["GET", "POST"])
def submit_messages():

    if request.method == "POST":
        control_messages_json = request.form.get("control-messages-json")
        publish_message(control_messages_json)
        data = {
            "status": "Successfully published task to kafka topic.",
            "status_code": 200,
            "control_messages": json.loads(control_messages_json)
        }
        data = json.dumps(data, indent=4)
        return data

    if request.method == "GET":
        return render_template("home.html")
