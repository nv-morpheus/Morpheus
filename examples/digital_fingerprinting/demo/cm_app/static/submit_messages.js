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
      convertToJson();
    });

  // Function to convert list of control messages to JSON object
  function convertToJson() {
  var inputsContainer = $('#inputs-container');
  var inputs = inputsContainer.find('.input');
  var jsonOutput = {};
  jsonOutput.inputs = [];

  inputs.each(function(index) {
     var input = $(this);
     var dataType = input.find('select[name="type"]').val();
     var metadataContainer = input.find('.metadata-container');
     var metadata = metadataContainer.find('.metadata');
     var metadataJson = {};

     // add metadata section
     metadata.each(function(index) {
        var metadataItem = $(this);
        var key = metadataItem.find('input[name="metadata-key"]').val();
        var dataType = metadataItem.find('select[name="metadata-type"]').val();
        var value = metadataItem.find('input[name="metadata-value"]').val();

        if (dataType === "text-array")
              value = value.split(",");
        if (dataType === "Number")
              value = parseInt(value)
        if (dataType === "dict")
              value = JSON.parse(value)

        metadataJson[key] = value;
     });

     var tasksContainer = input.find('.tasks-container');
     var tasks = tasksContainer.find('.task');
     var tasksJson = [];

     // add tasks section
     tasks.each(function(index) {
        var task = $(this);
        var taskType = task.find('select[name="task-type"]').val();
        var propertiesContainer = task.find('.properties-container');
        var properties = propertiesContainer.find('.property');
        var propertiesJson = {};

        properties.each(function(index) {
        var property = $(this);
        var key = property.find('input[name="property-key"]').val();
        var dataType = property.find('select[name="property-type"]').val();
        var value = property.find('input[name="property-value"]').val();

        if (dataType === "text-array")
           value = value.split(",");

        if (dataType === "Number")
           value = parseInt(value)

        if (dataType === "dict")
           value = JSON.parse(value)

        propertiesJson[key] = value;
        });

        tasksJson.push({ "type": taskType, "properties": propertiesJson });
     });

     // add datatype to metadata
     metadataJson['data_type'] = dataType
     var inputJson = { "metadata": metadataJson, "tasks": tasksJson };
     jsonOutput.inputs.push(inputJson);
  });

  var jsonString = JSON.stringify(jsonOutput, null, 2);
  $('#control-messages-json').val(jsonString);
  }



    // Add new control message button functionality
    $("#add-input-btn").click(function() {
      var inputHtml = `
      <div class="input">
        <label>Type:</label>
        <select name="type">
          <option value="payload">Payload</option>
          <option value="streaming">Streaming</option>
        </select>
        <button type="button" class="add-metadata-btn">Add Metadata</button>
        <div class="metadata-container">
          <!-- Metadata will be dynamically added here -->
        </div>
        <div class="tasks-container">
          <!-- Tasks will be dynamically added here -->
        </div>
        <button type="button" class="add-task-btn">Add Task</button>
        <button type="button" class="remove-input-btn">Remove Control Message</button>
      </div>`;
      $("#inputs-container").append(inputHtml);
    });

    // Add new task button functionality using event delegation
    $("#inputs-container").on("click", ".add-task-btn", function() {
      var taskHtml = `
      <div class="task">
        <label>Type:</label>
        <select name="task-type">
          <option value="load">Load</option>
          <option value="inference">Inference</option>
          <option value="training">Training</option>
        </select>
        <button type="button" class="add-property-btn">Add Property</button>
        <div class="properties-container">
          <!-- Properties will be dynamically added here -->
        </div>
        <button type="button" class="remove-task-btn">Remove Task</button>
      </div>`;
      $(this).parent().find(".tasks-container").append(taskHtml);
    });

    // Add new property button functionality
    $("#inputs-container").on("click", ".add-property-btn", function() {
      var propertyHtml = `
      <div class="property">
        <input type="text" name="property-key" placeholder="key">
        <select name="property-type">
        <option value="text">DataType</option>
          <option value="text">Text</option>
          <option value="number">Number</option>
          <option value="boolean">Boolean</option>
          <option value="date">Date</option>
          <option value="text-array">Array</option>
          <option value="dict">Object</option>
        </select>
        <input type="text" name="property-value" placeholder="value">
        <button type="button" class="remove-property-btn">Remove</button>
      </div>`;
      $(this).siblings(".properties-container").append(propertyHtml);
    });

    $("#inputs-container").on("click", ".add-metadata-btn", function() {
      var metadataHtml = `
      <div class="metadata">
        <input type="text" name="metadata-key" placeholder="key">
        <select name="metadata-type">
          <option value="text">DataType</option>
          <option value="text">Text</option>
          <option value="number">Number</option>
          <option value="boolean">Boolean</option>
          <option value="date">Date</option>
          <option value="text-array">Array</option>
          <option value="dict">Object</option>
        </select>
        <input type="text" name="metadata-value" placeholder="key">
        <button type="button" class="remove-metadata-btn">Remove</button>
      </div>`;
      $(this).siblings(".metadata-container").append(metadataHtml);
    });

    // Remove input button functionality using event delegation
    $("#inputs-container").on("click", ".remove-input-btn", function() {
       $(this).parent().remove();
     });

     // Remove task button functionality using event delegation
     $("#inputs-container").on("click", ".remove-task-btn", function() {
       $(this).parent().remove();
     });

     // Remove property button functionality using event delegation
     $("#inputs-container").on("click", ".remove-property-btn", function() {
       $(this).parent().remove();
     });

     // Remove metadata button functionality using event delegation
     $("#inputs-container").on("click", ".remove-metadata-btn", function() {
       $(this).parent().remove();
     });

  });
