<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Sensitive Information Detection with Natural Language Processing (NLP) Example

This example illustrates how to use Morpheus to automatically detect Sensitive Information (SI) in network packets by utilizing a Natural Language Processing (NLP) neural network and Triton Inference Server.

## Background

The goal of this example is to identify potentially sensitive information in network packet data as quickly as possible to limit exposure and take corrective action. Sensitive information is a broad term but can be generalized to any data that should be guarded from unautorized access. Credit card numbers, passwords, authorization keys, and emails are all examples of sensitive information.

In this example, we will be using Morpheus' provided NLP SI Detection model. This model is capable of detecting 10 different categories of sensitive information:

1. Addresses
2. Bank Account Numbers
3. Credit Card Numbers
4. Email Addresses
5. Government ID Numbers
6. Personal Names
7. Passwords
8. Phone Numbers
9. Secret Keys (a.k.a Private Keys)
10. User IDs

### The Dataset

The dataset that this workflow was designed to process is PCAP, or Packet Capture data, that is serialized into a JSON format. Several different applications are capable of capurting this type of network traffic. Each packet contains information about the source, destination, timestamp, and body of the packet, among other things. For example, below is a single packet that is from a HTTP POST request to cumulusnetworks.com:

```json
{
  "timestamp": 1616380971990,
  "host_ip": "10.188.40.56",
  "data_len": "309",
  "data": "POST /simpledatagen/ HTTP/1.1\r\nHost: echo.gtc1.netqdev.cumulusnetworks.com\r\nUser-Agent: python-requests/2.22.0\r\nAccept-Encoding: gzip, deflate\r\nAccept: */*\r\nConnection: keep-alive\r\nContent-Length: 73\r\nContent-Type: application/json\r\n\r\n",
  "src_mac": "04:3f:72:bf:af:74",
  "dest_mac": "b4:a9:fc:3c:46:f8",
  "protocol": "6",
  "src_ip": "10.20.16.248",
  "dest_ip": "10.244.0.59",
  "src_port": "50410",
  "dest_port": "80",
  "flags": "24"
}
```

In this example, we will be using a simulated PCAP dataset that is known to contain SI from each of the 10 categories the model was trained for. The dataset is located at `data/pcap_dump.jsonlines`. The dataset is in the `.jsonlines` format which means each new line represents an new JSON object. In order to parse this data, it must be ingested, split by lines into individual JSON objects, and parsed. This will all be handled by Morpheus.

## Pipeline Architecture

The pipeline we will be using in this example is a simple feed-forward linear pipeline where the data from each stage flows on to the next. Simple linear pipelines with no custom stages, like this example, can be configured via the Morpheus CLI or using the Python library. In this example we will be using the Morpheus CLI.

Below is a visualization of the pipeline showing all of the stages and data types as it flows from one stage to the next.

![Pipeline](pipeline.png)


## Setup

This example utilizes the Triton Inference Server to perform inference. The neural network model is provided in a separate submodule repo. Make sure the Morpheus Models repo is checked out with:

```bash
git submodule update --init --recursive
```

### Launching Triton

From the Morpheus repo root directory, run the following to launch Triton and load the `sid-minibert` model:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:21.03-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model sid-minibert-onnx
```

Where `21.03` can be replaced with the current year and month of the Triton version to use. For example, to use May 2021, specify `nvcr.io/nvidia/tritonserver:21.05-py3`. Ensure that the version of TensorRT that is used in Triton matches the version of TensorRT elsewhere (see [NGC Deep Learning Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)).

This will launch Triton and only load the `sid-minibert-onnx` model. This model has been configured with a max batch size of 32, and to use dynamic batching for increased performance.

Once Triton has loaded the model, you should see the following in the output:

```
+-------------------+---------+--------+
| Model             | Version | Status |
+-------------------+---------+--------+
| sid-minibert-onnx | 1       | READY  |
+-------------------+---------+--------+
```


## Running the Pipeline

With the Morpheus CLI, an entire pipeline can be configured and run without writing any code. Using the `morpheus run pipeline-nlp` command, we can build the pipeline by specifying each stage's name and configuration right on the command line. The output of each stage will become the input for the next.

The following command line is the entire command to build and launch the pipeline. Each new line represents a new stage. The comment above each stage gives information about why the stage was added and configured this way.

```bash
export MORPHEUS_ROOT=../..
# Launch Morpheus printing debug messages
morpheus --debug --log_level=DEBUG \
   `# Run a pipeline with 8 threads and a model batch size of 32 (Must match Triton config)` \
   run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=32 \
   `# Specify a NLP pipeline with 256 sequence length (Must match Triton config)` \
   pipeline-nlp --model_seq_length=256 --viz_file ./pipeline.png \
   `# 1st Stage: Read from file` \
   from-file --filename=$MORPHEUS_ROOT/data/pcap_dump.jsonlines \
   `# 2nd Stage: Buffer upstream stage data (improves performance)` \
   buffer \
   `# 3rd Stage: Deserialize from JSON strings to objects` \
   deserialize \
   `# 4th Stage: Preprocessing converts the input data into BERT tokens` \
   preprocess --vocab_hash_file=$MORPHEUS_ROOT/data/bert-base-uncased-hash.txt --do_lower_case=True \
   `# 5th Stage: Another buffer before inference for performance` \
   buffer \
   `# 6th Stage: Send messages to Triton for inference. Specify the model loaded in Setup` \
   inf-triton --model_name=sid-minibert-onnx --server_url=localhost:8001 --force_convert_inputs=True \
   `# 7th Stage: Monitor stage prints throughput information to the console` \
   monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
   `# 8th Stage: Add results from inference to the messages` \
   add-class \
   `# 9th Stage: Filtering removes any messages that did not detect SI` \
   filter \
   `# 10th Stage: Convert from objects back into strings` \
   serialize --exclude '^ts_' \
   `# 11th Stage: Write out the JSON lines to the detections.jsonlines file` \
   to-file --filename=detections.jsonlines --overwrite
```

If successful, you should see the following output:

```bash
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
Config:
{
  "dask": {
    "use_processes": false
  },
  "debug": true,
  "feature_length": 256,
  "log_config_file": null,
  "log_level": 10,
  "mode": "NLP",
  "model_max_batch_size": 32,
  "num_threads": 8,
  "pipeline_batch_size": 1024,
  "use_dask": false
}
====Building Pipeline====
Added source: <from-file-0; FileSourceStage(filename=../../data/pcap_dump.jsonlines, iterative=None)>
  └─> cudf.DataFrame
Added stage: <buffer-1; BufferStage(count=1000)>
  └─ cudf.DataFrame -> cudf.DataFrame
Adding timestamp info for stage: 'deserialize'
Added stage: <deserialize-2; DeserializeStage()>
  └─ cudf.DataFrame -> morpheus.MultiMessage
Adding timestamp info for stage: 'preprocess-nlp'
Added stage: <preprocess-nlp-3; PreprocessNLPStage(vocab_hash_file=../../data/bert-base-uncased-hash.txt, truncation=False, do_lower_case=True, add_special_tokens=False, stride=-1)>
  └─ morpheus.MultiMessage -> morpheus.MultiInferenceNLPMessage
Added stage: <buffer-4; BufferStage(count=1000)>
  └─ morpheus.MultiInferenceNLPMessage -> morpheus.MultiInferenceNLPMessage
Adding timestamp info for stage: 'inference'
Added stage: <inference-5; TritonInferenceStage(model_name=sid-minibert-onnx, server_url=localhost:8001, force_convert_inputs=True)>
  └─ morpheus.MultiInferenceNLPMessage -> morpheus.MultiResponseProbsMessage
Added stage: <monitor-6; MonitorStage(description=Inference Rate, smoothing=0.001, unit=inf, determine_count_fn=None)>
  └─ morpheus.MultiResponseProbsMessage -> morpheus.MultiResponseProbsMessage
Added stage: <add-class-7; AddClassificationsStage(threshold=0.5, labels_file=None, labels=['address', 'bank_acct', 'credit_card', 'email', 'govt_id', 'name', 'password', 'phone_num', 'secret_keys', 'user'], prefix=si_)>
  └─ morpheus.MultiResponseProbsMessage -> morpheus.MultiResponseProbsMessage
Added stage: <filter-8; FilterDetectionsStage(threshold=0.5)>
  └─ morpheus.MultiResponseProbsMessage -> morpheus.MultiResponseProbsMessage
Added stage: <serialize-9; SerializeStage(include=[], exclude=['^ts_'], as_cudf_df=False)>
  └─ morpheus.MultiResponseProbsMessage -> List[str]
Added stage: <to-file-10; WriteToFileStage(filename=detections.jsonlines, overwrite=True)>
  └─ List[str] -> List[str]
====Building Pipeline Complete!====
Pipeline visualization saved to ./pipeline.png
====Starting Pipeline====
====Pipeline Started====
Inference Rate: 93085inf [01:50, 885.45inf/s]
```

**Note:** The pipeline will not shut down when complete. Once the number of inferences has stopped changing, press Ctrl+C to exit.

The output file `detections.jsonlines` will contain PCAP messages that contain some SI (any class with a predection greater that 0.5).
