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

In addition to the Morpheus pipeline settings, we also have a configuration file called [modules_conf.json](./resource/modules_conf.json) that is specific to modules. When using MRC SegmentModule, pipelines need this configuration file. Additional information is included in the [Morpheus Pipeline with Modules](../../../../../docs/source/developer_guide/guides/6_digital_fingerprinting_reference.md#morpheus-pipeline-with-modules)

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
-------------------------------------------------------------------------------------------------------- benchmark: 19 tests --------------------------------------------------------------------------------------------------------
Name (time in ms)                                          Min                    Max                   Mean              StdDev                 Median                 IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_only_load_e2e            226.3854 (1.0)         283.0055 (1.0)         259.3731 (1.0)       24.3098 (1.0)         269.2701 (1.0)       40.5459 (1.0)           1;0  3.8554 (1.0)           5           1
test_dfp_modules_duo_payload_inference_e2e            976.1599 (4.31)      1,147.7819 (4.06)      1,067.5186 (4.12)      65.2043 (2.68)      1,088.5716 (4.04)      86.9582 (2.14)          2;0  0.9368 (0.24)          5           1
test_dfp_stages_duo_inference_e2e                   1,040.1275 (4.59)      1,328.9118 (4.70)      1,158.5368 (4.47)     127.0640 (5.23)      1,127.6553 (4.19)     223.5278 (5.51)          1;0  0.8632 (0.22)          5           1
test_dfp_modules_azure_payload_inference_e2e        1,075.9931 (4.75)      1,313.8863 (4.64)      1,163.2758 (4.48)      90.5340 (3.72)      1,142.0053 (4.24)      95.3948 (2.35)          1;0  0.8596 (0.22)          5           1
test_dfp_stages_azure_inference_e2e                 1,102.1970 (4.87)      1,436.8655 (5.08)      1,243.6478 (4.79)     147.9676 (6.09)      1,164.8561 (4.33)     246.8259 (6.09)          1;0  0.8041 (0.21)          5           1
test_dfp_modules_duo_streaming_inference_e2e        1,261.8304 (5.57)      1,406.6397 (4.97)      1,333.9344 (5.14)      52.9789 (2.18)      1,324.8074 (4.92)      62.6631 (1.55)          2;0  0.7497 (0.19)          5           1
test_dfp_modules_azure_streaming_inference_e2e      1,332.5694 (5.89)      1,506.8211 (5.32)      1,415.3912 (5.46)      67.6594 (2.78)      1,417.5592 (5.26)     101.9428 (2.51)          2;0  0.7065 (0.18)          5           1
test_dfp_modules_duo_streaming_only_load_e2e        1,805.8288 (7.98)      2,354.6001 (8.32)      2,045.9313 (7.89)     199.3942 (8.20)      2,045.7892 (7.60)     202.2794 (4.99)          2;0  0.4888 (0.13)          5           1
test_dfp_modules_duo_payload_training_e2e           9,037.7003 (39.92)     9,836.9510 (34.76)     9,367.2792 (36.12)    330.3668 (13.59)     9,207.2873 (34.19)    502.7229 (12.40)         1;0  0.1068 (0.03)          5           1
test_dfp_modules_duo_payload_lti_e2e                9,954.3053 (43.97)    10,534.4838 (37.22)    10,247.6966 (39.51)    246.8732 (10.16)    10,224.6111 (37.97)    434.5221 (10.72)         2;0  0.0976 (0.03)          5           1
test_dfp_modules_azure_payload_training_e2e        11,542.1990 (50.98)    11,704.6100 (41.36)    11,625.2338 (44.82)     72.5717 (2.99)     11,648.4413 (43.26)    130.2369 (3.21)          2;0  0.0860 (0.02)          5           1
test_dfp_modules_azure_payload_lti_e2e             12,414.6397 (54.84)    13,634.3140 (48.18)    13,112.0041 (50.55)    492.8452 (20.27)    13,270.1088 (49.28)    763.9778 (18.84)         2;0  0.0763 (0.02)          5           1
test_dfp_stages_duo_training_e2e                   15,892.6129 (70.20)    16,538.2125 (58.44)    16,301.0573 (62.85)    242.4913 (9.98)     16,351.5376 (60.73)    212.1910 (5.23)          1;1  0.0613 (0.02)          5           1
test_dfp_modules_duo_streaming_training_e2e        27,783.2057 (122.73)   28,387.4788 (100.31)   27,956.0751 (107.78)   249.0318 (10.24)    27,853.2863 (103.44)   253.7971 (6.26)          1;0  0.0358 (0.01)          5           1
test_dfp_stages_azure_training_e2e                 28,264.0585 (124.85)   29,443.4046 (104.04)   28,879.5257 (111.34)   476.5615 (19.60)    28,900.8030 (107.33)   781.7848 (19.28)         2;0  0.0346 (0.01)          5           1
test_dfp_modules_duo_streaming_payload_e2e         29,466.8204 (130.16)   30,338.3991 (107.20)   29,855.7080 (115.11)   377.8633 (15.54)    29,864.8365 (110.91)   669.7878 (16.52)         2;0  0.0335 (0.01)          5           1
test_dfp_modules_duo_streaming_lti_e2e             30,443.9077 (134.48)   31,385.2542 (110.90)   30,875.1344 (119.04)   334.9455 (13.78)    30,853.1295 (114.58)   258.4034 (6.37)          2;1  0.0324 (0.01)          5           1
test_dfp_modules_azure_streaming_training_e2e      51,950.9638 (229.48)   52,498.6271 (185.50)   52,257.2178 (201.48)   259.6411 (10.68)    52,317.4839 (194.29)   494.9443 (12.21)         1;0  0.0191 (0.00)          5           1
test_dfp_modules_azure_streaming_lti_e2e           54,148.7980 (239.19)   54,953.7450 (194.18)   54,525.3318 (210.22)   313.7135 (12.90)    54,540.2730 (202.55)   473.5052 (11.68)         2;0  0.0183 (0.00)          5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
