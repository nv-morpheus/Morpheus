/*
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

$(document).ready(function() {

    $("#submit").click(function() {
       submitForm();
    });

  // Function to convert inputs-container and child data to JSON
  function submitForm() {
    // get all the input fields in the inputs-container div
    const inputs = $('#inputs-container :input');

    // create an empty object to hold the field values
    const formData = {};

    // loop through the inputs and add their values to the formData object
    inputs.each(function() {
       formData[this.name] = $(this).val();
    });

    let loadTask = {"type": "load",
                 "properties": {
                 "loader_id": "fsspec",
                 "files": formData["files"].split(","),
               }};
     let trainingTask = {
       "type": "training",
       "properties": {}
     };

     let tasks = [loadTask, trainingTask];
     let samplingRate = parseInt(formData["sampling_rate_s"]);

     let batching_options ={}
     batching_options["period"] = formData["period"];
     batching_options["sampling_rate_s"] = samplingRate;
     batching_options["start_time"] = formData["start_time"];
     batching_options["end_time"] = formData["end_time"];

     let metadata = {
     "data_type": "payload",
     "batching_options": batching_options
   };

   let controlMessage = {"inputs": [{"tasks": tasks, "metadata": metadata}]};

   // Submit form as JSON
   var jsonString = JSON.stringify(controlMessage, null, 2);

   $('#control-messages-json').val(jsonString);

  }

  });

  function checkDateTime() {
    var startDate = new Date(document.getElementById("start_time").value);
    var endDate = new Date(document.getElementById("end_time").value);
    if (startDate > endDate) {
       alert("Start time cannot be greater than end time");
       return false;
    }
    return true;
 }
