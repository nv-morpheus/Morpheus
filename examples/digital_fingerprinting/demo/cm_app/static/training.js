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
   console.error(jsonString);
   $('#control-messages-json').val(jsonString);
 
  }
 
  });