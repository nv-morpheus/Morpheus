<!--
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

# Running DFP E2E Benchmarks

### Set Environment

To set up and run the benchmarks on production DFP pipeline, follow the instructions provided [here](../../README.md). Once the Morpheus container and the MLFlow server have been set up and running with `docker-compose`. Attach to the Morpheus pipeline container and download the sample data from S3 per the document's instructions.

## Requirements
> **Note**: Make sure `gputil`, `dask` and `distributed` are installed in your Conda environment before running the benchmarks. Run the installation command specified below if not.

```bash
conda install gputil dask==2022.7.0 distributed==2022.7.0
```

### Run E2E Benchmarks

Benchmarks are run using `pytest-benchmark`. By default, there are five rounds of measurement. For each round, there will be one iteration of each workflow. Measurements are taken for each round. Final results such as `min`, `max` and `mean` times will be based on these measurements.

To provide your own calibration or use other `pytest-benchmark` features with these workflows, please refer to their [documentation](https://pytest-benchmark.readthedocs.io/en/latest/).

Morpheus pipeline configurations for each workflow are managed using [pipelines_conf.json](./resource/pipelines_conf.json). For example, this is the Morpheus configuration for  `duo_training_modules`:
```
"test_dfp_modules_azure_training_e2e": {
      "message_path": "./resource/control_message_azure_training.json",
      "num_threads": 12,
      "pipeline_batch_size": 256,
      "edge_buffer_size": 128,
      "start_time": "2022-08-01",
      "duration": "60d"
},
...
```

In addition to the Morpheus pipeline settings, we also have a configuration file called [modules_conf.json](./resource/modules_conf.json) that is specific to modules. When using MRC SegmentModule, pipelines need this configuration file. Additional information is included in the [Morpheus Pipeline with Modules](../../../../../docs/source/developer_guide/guides/6_digital_fingerprinting_reference.md#morpheus-pipeline-with-modules)

Benchmarks for an individual workflow can be run using the following:

```

pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py::<test-workflow>
```
The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

The `--benchmark-warmup` and `--benchmark-warmup-iterations` options are used to run the workflow(s) once before starting measurements. This is because, if it does not already exist, the preprocessed data is cached during the initial run.

`<test-workflow>` is the name of the test to run benchmarks on. This can be one of the following:
- `test_dfp_modules_azure_payload_inference_e2e`
- `test_dfp_modules_azure_payload_lti_e2e`
- `test_dfp_modules_azure_payload_training_e2e`
- `test_dfp_modules_azure_streaming_inference_e2e`
- `test_dfp_modules_azure_streaming_lti_e2e`
- `test_dfp_modules_azure_streaming_training_e2e`
- `test_dfp_modules_duo_payload_inference_e2e`
- `test_dfp_modules_duo_payload_lti_e2e`
- `test_dfp_modules_duo_payload_only_load_e2e`
- `test_dfp_modules_duo_payload_training_e2e`
- `test_dfp_modules_duo_streaming_inference_e2e`
- `test_dfp_modules_duo_streaming_lti_e2e`
- `test_dfp_modules_duo_streaming_only_load_e2e`
- `test_dfp_modules_duo_streaming_payload_e2e`
- `test_dfp_modules_duo_streaming_training_e2e`
- `test_dfp_stages_azure_training_e2e`
- `test_dfp_stages_azure_inference_e2e`
- `test_dfp_stages_duo_training_e2e`
- `test_dfp_stages_duo_inference_e2e`

For example, to run E2E benchmarks on the DFP training (modules) workflow on the duo logs:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py::test_dfp_modules_azure_payload_lti_e2e
```

To run E2E benchmarks on all workflows:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py
```

The console output should look like this:
```
------------------------------------------------------------------------------------------------------- benchmark: 19 tests -------------------------------------------------------------------------------------------------------
Name (time in ms)                                          Min                    Max                   Mean             StdDev                 Median                IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_only_load_e2e            221.7548 (1.0)         313.8652 (1.0)         263.8946 (1.0)      35.5942 (inf)         251.0703 (1.0)      49.3962 (inf)           2;0  3.7894 (1.0)           5           1
test_dfp_modules_duo_payload_inference_e2e          1,010.4983 (4.56)      1,010.4983 (3.22)      1,010.4983 (3.83)      0.0000 (1.0)       1,010.4983 (4.02)      0.0000 (1.0)           0;0  0.9896 (0.26)          1           1
test_dfp_modules_azure_payload_inference_e2e        1,160.3311 (5.23)      1,160.3311 (3.70)      1,160.3311 (4.40)      0.0000 (1.0)       1,160.3311 (4.62)      0.0000 (1.0)           0;0  0.8618 (0.23)          1           1
test_dfp_stages_duo_inference_e2e                   1,221.0156 (5.51)      1,221.0156 (3.89)      1,221.0156 (4.63)      0.0000 (1.0)       1,221.0156 (4.86)      0.0000 (1.0)           0;0  0.8190 (0.22)          1           1
test_dfp_stages_azure_inference_e2e                 1,462.4917 (6.60)      1,462.4917 (4.66)      1,462.4917 (5.54)      0.0000 (1.0)       1,462.4917 (5.83)      0.0000 (1.0)           0;0  0.6838 (0.18)          1           1
test_dfp_modules_azure_streaming_inference_e2e      1,562.7886 (7.05)      1,562.7886 (4.98)      1,562.7886 (5.92)      0.0000 (1.0)       1,562.7886 (6.22)      0.0000 (1.0)           0;0  0.6399 (0.17)          1           1
test_dfp_modules_duo_streaming_inference_e2e        1,626.7846 (7.34)      1,626.7846 (5.18)      1,626.7846 (6.16)      0.0000 (1.0)       1,626.7846 (6.48)      0.0000 (1.0)           0;0  0.6147 (0.16)          1           1
test_dfp_modules_duo_payload_training_e2e           9,909.2326 (44.69)     9,909.2326 (31.57)     9,909.2326 (37.55)     0.0000 (1.0)       9,909.2326 (39.47)     0.0000 (1.0)           0;0  0.1009 (0.03)          1           1
test_dfp_modules_duo_payload_lti_e2e               11,283.7325 (50.88)    11,283.7325 (35.95)    11,283.7325 (42.76)     0.0000 (1.0)      11,283.7325 (44.94)     0.0000 (1.0)           0;0  0.0886 (0.02)          1           1
test_dfp_modules_azure_payload_training_e2e        12,097.5285 (54.55)    12,097.5285 (38.54)    12,097.5285 (45.84)     0.0000 (1.0)      12,097.5285 (48.18)     0.0000 (1.0)           0;0  0.0827 (0.02)          1           1
test_dfp_modules_azure_payload_lti_e2e             13,467.1761 (60.73)    13,467.1761 (42.91)    13,467.1761 (51.03)     0.0000 (1.0)      13,467.1761 (53.64)     0.0000 (1.0)           0;0  0.0743 (0.02)          1           1
test_dfp_stages_duo_training_e2e                   18,871.9930 (85.10)    18,871.9930 (60.13)    18,871.9930 (71.51)     0.0000 (1.0)      18,871.9930 (75.17)     0.0000 (1.0)           0;0  0.0530 (0.01)          1           1
test_dfp_stages_azure_training_e2e                 30,399.7126 (137.09)   30,399.7126 (96.86)    30,399.7126 (115.20)    0.0000 (1.0)      30,399.7126 (121.08)    0.0000 (1.0)           0;0  0.0329 (0.01)          1           1
test_dfp_modules_duo_streaming_payload_e2e         33,018.3594 (148.90)   33,018.3594 (105.20)   33,018.3594 (125.12)    0.0000 (1.0)      33,018.3594 (131.51)    0.0000 (1.0)           0;0  0.0303 (0.01)          1           1
test_dfp_modules_duo_streaming_training_e2e        33,672.9700 (151.85)   33,672.9700 (107.28)   33,672.9700 (127.60)    0.0000 (1.0)      33,672.9700 (134.12)    0.0000 (1.0)           0;0  0.0297 (0.01)          1           1
test_dfp_modules_duo_streaming_only_load_e2e       35,410.0752 (159.68)   35,410.0752 (112.82)   35,410.0752 (134.18)    0.0000 (1.0)      35,410.0752 (141.04)    0.0000 (1.0)           0;0  0.0282 (0.01)          1           1
test_dfp_modules_duo_streaming_lti_e2e             36,251.7741 (163.48)   36,251.7741 (115.50)   36,251.7741 (137.37)    0.0000 (1.0)      36,251.7741 (144.39)    0.0000 (1.0)           0;0  0.0276 (0.01)          1           1
test_dfp_modules_azure_streaming_training_e2e      54,888.6326 (247.52)   54,888.6326 (174.88)   54,888.6326 (207.99)    0.0000 (1.0)      54,888.6326 (218.62)    0.0000 (1.0)           0;0  0.0182 (0.00)          1           1
test_dfp_modules_azure_streaming_lti_e2e           57,296.2454 (258.38)   57,296.2454 (182.55)   57,296.2454 (217.12)    0.0000 (1.0)      57,296.2454 (228.21)    0.0000 (1.0)           0;0  0.0175 (0.00)          1           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Benchmarks Report

Each time you run the benchmarks as above, a comprehensive report for each run will be generated and saved to a JSON file in  `./benchmarks/.benchmarks`. The file name will begin
with `000N` where N is incremented for every run. For example, the report file name for first benchmarks run will look like:
```
0001_f492e1952d5981527d89229e557006a1db992e5f_20230201_230822.json
```

A hook to `pytest-benchmark` was developed to add the following information to the JSON report:

GPU(s) used by Morpheus. For example:
```
"gpu_0": {
    "id": 0,
    "name": "Quadro RTX 8000",
    "load": "0.0%",
    "free_memory": "47627.0MB",
    "used_memory": "965.0MB",
    "temperature": "55.0 C",
    "uuid": "GPU-6fa37f47-763b-fc49-1b15-75b0d36525bf"
}
```

Morpheus config for each workflow:
- num_threads
- pipeline_batch_size
- edge_buffer_size
- start_time
- duration

Additional benchmark stats for each workflow:
- input_lines
- min_throughput_lines
- max_throughput_lines
- mean_throughput_lines
- median_throughput_lines
- input_bytes
- min_throughput_bytes
- max_throughput_bytes
- mean_throughput_bytes
- median_throughput_bytes
