from flask import render_template
from flask import request
from cm_app.helper import publish_message
from cm_app.helper import process_cm

from . import app


@app.route('/', methods=["GET", "POST"])
def submit_messages():

    if request.method == "POST":
        return process_cm(request)

    if request.method == "GET":
        return render_template("submit_messages.html")

@app.route('/training', methods=["GET", "POST"])
def training():

    if request.method == "POST":
        return process_cm(request)

    if request.method == "GET":
        return render_template("training.html")
