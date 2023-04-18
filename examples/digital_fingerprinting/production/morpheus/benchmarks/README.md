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

Morpheus pipeline configurations for each workflow are managed using [pipelines_conf.json](./resource/pipelines_conf.json). For example, this is the Morpheus configuration for  `dfp_modules_duo_payload_inference`:
```
"test_dfp_modules_duo_payload_inference_e2e": {
		"message_path": "./resource/control_messages/duo_payload_inference.json",
		"num_threads": 12,
		"pipeline_batch_size": 256,
		"edge_buffer_size": 128,
		"start_time": "2022-08-01",
		"duration": "60d",
		"userid_column_name": "username",
		"timestamp_column_name": "timestamp",
		"source": "duo",
		"use_cpp": true
},
...
```

When using MRC SegmentModule, pipelines need requires module configuration which does gets generated within the test. Additional information is included in the [Morpheus Pipeline with Modules](../../../../../docs/source/developer_guide/guides/6_digital_fingerprinting_reference.md#morpheus-pipeline-with-modules)

To ensure that the [file_to_df_loader.py](../../../../../morpheus/loaders/file_to_df_loader.py) utilizes the same type of downloading mechanism, set `MORPHEUS FILE DOWNLOAD TYPE` environment variable with any one of given choices (`multiprocess`, `dask`, `dask thread`, `single thread`).

```
export MORPHEUS_FILE_DOWNLOAD_TYPE=multiprocess
```

Benchmarks for an individual workflow can be run using the following:

```

pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py::<test-workflow>
```
The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

The `--benchmark-warmup` and `--benchmark-warmup-iterations` options are used to run the workflow(s) once before starting measurements. This is because, if it does not already exist, the preprocessed data is cached during the initial run.

`<test-workflow>` is the name of the test to run benchmarks on. This can be one of the following:
- `test_dfp_modules_azure_payload_inference_e2e`
- `test_dfp_modules_azure_payload_lti_e2e`
- `test_dfp_modules_azure_payload_lti_s3_e2e`
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

For example, to run E2E benchmarks on the DFP training (modules) workflow on the azure logs:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py::test_dfp_modules_azure_payload_lti_e2e
```

To run E2E benchmarks on all workflows:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py
```

#### Training (Azure):
```bash
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py -k 'test_dfp_modules_azure_payload_training_e2e or test_dfp_stages_azure_training_e2e or test_dfp_modules_azure_streaming_training_e2e'
```

Output:
```
---------------------------------------------------------------------------------------------- benchmark: 3 tests ----------------------------------------------------------------------------------------------
Name (time in s)                                      Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_payload_training_e2e        8.5839 (1.0)      10.0795 (1.0)       9.2604 (1.0)      0.6213 (1.0)       9.4125 (1.0)      1.0004 (1.0)           2;0  0.1080 (1.0)           5           1
test_dfp_stages_azure_training_e2e                29.0398 (3.38)     31.3069 (3.11)     30.4112 (3.28)     0.9112 (1.47)     30.4538 (3.24)     1.3482 (1.35)          1;0  0.0329 (0.30)          5           1
test_dfp_modules_azure_streaming_training_e2e     34.8974 (4.07)     38.3883 (3.81)     37.0896 (4.01)     1.7385 (2.80)     38.2802 (4.07)     3.0295 (3.03)          1;0  0.0270 (0.25)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```


#### Inference (Azure):
```bash
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py -k 'test_dfp_modules_azure_payload_inference_e2e or test_dfp_stages_azure_inference_e2e or test_dfp_modules_azure_streaming_inference_e2e'
```

Output:
```
--------------------------------------------------------------------------------------------- benchmark: 3 tests --------------------------------------------------------------------------------------------
Name (time in s)                                      Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_streaming_inference_e2e     1.2708 (1.0)      1.4836 (1.0)      1.3580 (1.0)      0.0792 (1.0)      1.3398 (1.0)      0.0898 (1.0)           2;0  0.7364 (1.0)           5           1
test_dfp_modules_azure_payload_inference_e2e       1.3115 (1.03)     1.5259 (1.03)     1.4215 (1.05)     0.0954 (1.20)     1.4078 (1.05)     0.1747 (1.95)          2;0  0.7035 (0.96)          5           1
test_dfp_stages_azure_inference_e2e                1.5362 (1.21)     1.8836 (1.27)     1.6827 (1.24)     0.1455 (1.84)     1.6410 (1.22)     0.2406 (2.68)          2;0  0.5943 (0.81)          5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Training (Duo)
```bash
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py -k 'test_dfp_modules_duo_payload_training_e2e or test_dfp_stages_duo_training_e2e or test_dfp_modules_duo_streaming_training_e2e'
```

Output:
```
--------------------------------------------------------------------------------------------- benchmark: 3 tests ---------------------------------------------------------------------------------------------
Name (time in s)                                    Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_training_e2e        8.2557 (1.0)       8.8273 (1.0)       8.5059 (1.0)      0.2284 (1.0)       8.4911 (1.0)      0.3575 (1.0)           2;0  0.1176 (1.0)           5           1
test_dfp_stages_duo_training_e2e                19.5853 (2.37)     22.5840 (2.56)     21.4216 (2.52)     1.2993 (5.69)     21.9340 (2.58)     2.1545 (6.03)          1;0  0.0467 (0.40)          5           1
test_dfp_modules_duo_streaming_training_e2e     22.0140 (2.67)     24.0175 (2.72)     23.0668 (2.71)     0.8957 (3.92)     23.3231 (2.75)     1.6232 (4.54)          2;0  0.0434 (0.37)          5           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Inference (Duo)
```bash
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py -k 'test_dfp_modules_duo_payload_inference_e2e or test_dfp_stages_duo_inference_e2e or test_dfp_modules_duo_streaming_inference_e2e'
```

Output:
```
----------------------------------------------------------------------------------------------------- benchmark: 3 tests -----------------------------------------------------------------------------------------------------
Name (time in ms)                                       Min                   Max                  Mean             StdDev                Median                 IQR            Outliers     OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_inference_e2e         875.6646 (1.0)        929.5996 (1.0)        906.9823 (1.0)      22.7483 (1.0)        906.2029 (1.0)       37.9495 (1.0)           1;0  1.1026 (1.0)           5           1
test_dfp_modules_duo_streaming_inference_e2e       967.9455 (1.11)     1,046.5953 (1.13)     1,006.8804 (1.11)     33.0146 (1.45)     1,016.7610 (1.12)      55.0353 (1.45)          2;0  0.9932 (0.90)          5           1
test_dfp_stages_duo_inference_e2e                1,086.2484 (1.24)     1,222.0949 (1.31)     1,146.0136 (1.26)     59.7578 (2.63)     1,123.3352 (1.24)     104.2315 (2.75)          2;0  0.8726 (0.79)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Integrated Training and Inference (Azure)
```bash
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py -k 'test_dfp_modules_azure_payload_lti_e2e or test_dfp_modules_azure_streaming_lti_e2e'
```

Output:
```
-------------------------------------------------------------------------------------------- benchmark: 2 tests -------------------------------------------------------------------------------------------
Name (time in s)                                 Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_payload_lti_e2e       10.1342 (1.0)      11.7124 (1.0)      10.9306 (1.0)      0.6102 (1.04)     10.8310 (1.0)      0.9002 (1.79)          2;0  0.0915 (1.0)           5           1
test_dfp_modules_azure_streaming_lti_e2e     31.8959 (3.15)     33.2688 (2.84)     32.9316 (3.01)     0.5857 (1.0)      33.1723 (3.06)     0.5040 (1.0)           1;1  0.0304 (0.33)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Integrated Training and Inference (Duo)
```bash
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py -k 'test_dfp_modules_duo_payload_lti_e2e or test_dfp_modules_duo_streaming_lti_e2e'
```

Output:
```
------------------------------------------------------------------------------------------- benchmark: 2 tests ------------------------------------------------------------------------------------------
Name (time in s)                               Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_lti_e2e        7.8609 (1.0)       8.0143 (1.0)       7.9721 (1.0)      0.0632 (1.0)       8.0007 (1.0)      0.0533 (1.0)           1;1  0.1254 (1.0)           5           1
test_dfp_modules_duo_streaming_lti_e2e     20.6080 (2.62)     22.0563 (2.75)     21.4089 (2.69)     0.5599 (8.86)     21.3826 (2.67)     0.8076 (15.15)         2;0  0.0467 (0.37)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Integrated Training and Inference from s3 (Azure)
```bash
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_dfp_pipeline.py::test_dfp_modules_azure_payload_lti_s3_e2e
```

Output:
```
---------------------------------------------------------- benchmark: 1 tests ----------------------------------------------------------
Name (time in s)                                  Min      Max     Mean   StdDev   Median      IQR  Outliers     OPS  Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_payload_lti_s3_e2e     44.9810  74.9993  59.6432  10.8338  58.9347  12.0293       2;0  0.0168       5           1
----------------------------------------------------------------------------------------------------------------------------------------
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
- userid_column_name
- timestamp_column_name
- source
- use_cpp

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
