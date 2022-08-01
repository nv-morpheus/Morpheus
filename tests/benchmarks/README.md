<!--
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Running E2E Benchmarks

### Set up Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Docker image from NGC (https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) suitable for your environment.

Example:

```
docker pull nvcr.io/nvidia/tritonserver:22.02-py3
```

##### Start Triton Inference Server container
```
cd ${MORPHEUS_ROOT}/models

docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD:/models nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=/models/triton-model-repo --model-control-mode=explicit --load-model sid-minibert-onnx --load-model abp-nvsmi-xgb --load-model phishing-bert-onnx
```

##### Verify Model Deployments
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```
+--------------------+---------+--------+
| Model              | Version | Status |
+--------------------+---------+--------+
| abp-nvsmi-xgb      | 1       | READY  |
| phishing-bert-onnx | 1       | READY  |
| sid-minibert-onnx  | 1       | READY  |
+--------------------+---------+--------+
```

### Run E2E Benchmarks

Benchmarks are run using `pytest-benchmark`. Benchmarks for an individual workflow can be run using the following:

```
cd tests/benchmarks

pytest -s --benchmark-enable --benchmark-autosave test_bench_e2e_pipelines.py::<test-workflow>
```
The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

`<test-workflow>` is the name of the test to run benchmarks on. This can be `test_sid_nlp_e2e`, `test_abp_fil_e2e`, `test_phishing_nlp_e2e` or `test_cloudtrail_ae_e2e`.

For example, to run E2E benchmarks on the SID NLP workflow:
```
pytest -s --benchmark-enable --benchmark-autosave test_bench_e2e_pipelines.py::test_sid_nlp_e2e
```

To run E2E benchmarks on all workflows:
```
pytest -s --benchmark-enable --benchmark-autosave test_bench_e2e_pipelines.py
```

The console output should look like this:
```
------------------------------------------------------------------------------------------- benchmark: 4 tests ------------------------------------------------------------------------------------------
Name (time in ms)                 Min                   Max                  Mean              StdDev                Median                 IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_phishing_nlp_e2e        834.5413 (1.0)        892.8774 (1.0)        858.9724 (1.0)       22.5832 (1.0)        854.7082 (1.0)       31.7465 (1.0)           2;0  1.1642 (1.0)           5           1
test_sid_nlp_e2e           2,055.0733 (2.46)     2,118.1255 (2.37)     2,095.8951 (2.44)      26.2586 (1.16)     2,105.8771 (2.46)      38.5301 (1.21)          1;0  0.4771 (0.41)          5           1
test_abp_fil_e2e           5,016.7639 (6.01)     5,292.9841 (5.93)     5,179.0901 (6.03)     121.5466 (5.38)     5,195.2253 (6.08)     215.2213 (6.78)          1;0  0.1931 (0.17)          5           1
test_cloudtrail_ae_e2e     6,929.7436 (8.30)     7,157.0487 (8.02)     6,995.1969 (8.14)      92.8935 (4.11)     6,971.9611 (8.16)      87.2056 (2.75)          1;1  0.1430 (0.12)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

A comprehensive report for each test run will be saved to a JSON file in  `./tests/benchmarks/.benchmarks`. This will include throughput (lines/sec, bytes/sec), GPU info and Morpheus configs for each test workflow.
