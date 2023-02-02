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


### Run E2E Benchmarks

Benchmarks are run using `pytest-benchmark`. By default, there are five rounds of measurement. For each round, there will be one iteration of each workflow. Measurements are taken for each round. Final results such as `min`, `max` and `mean` times will be based on these measurements.

To provide your own calibration or use other `pytest-benchmark` features with these workflows, please refer to their [documentation](https://pytest-benchmark.readthedocs.io/en/latest/).

Morpheus pipeline configurations for each workflow are managed using [pipelines_conf.json](./resource/pipelines_conf.json). For example, this is the Morpheus configuration for  `duo_training_modules`:
```
"dfp_training_duo_modules_e2e": {
    "file_path": "../../../../data/dfp/duo-training-data/*.json",
    "num_threads": 8,
    "pipeline_batch_size": 1024,
    "edge_buffer_size": 4,
    "start_time": "2022-08-01",
    "duration": "60d"
},
...
```

In addition to the Morpheus pipeline settings, we also have a configuration file called [modules_conf.json](./resource/modules_conf.json) that is specific to modules. When using MRC SegmentModule, pipelines need this configuration file. Additional information is included in the [Morpheus Pipeline with Modules](../../../../../docs/source/developer_guide/guides/6_digital_fingerprinting_reference.md#morpheus-pipeline-with-modules)

Benchmarks for an individual workflow can be run using the following:

```

pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave bench_e2e_dfp_training_pipeline.py::<test-workflow>
```
The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

The `--benchmark-warmup` and `--benchmark-warmup-iterations` options are used to run the workflow(s) once before starting measurements. This is because, if it does not already exist, the preprocessed data is cached during the initial run.

`<test-workflow>` is the name of the test to run benchmarks on. This can be one of the following:
- `test_dfp_training_duo_modules_e2e`
- `test_dfp_training_duo_stages_e2e`
- `test_dfp_training_azure_modules_e2e`
- `test_dfp_training_azure_stages_e2e`

For example, to run E2E benchmarks on the DFP training (modules) workflow on the duo logs:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave bench_e2e_dfp_training_pipeline.py::test_dfp_training_duo_modules_e2e
```

To run E2E benchmarks on all workflows:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave bench_e2e_dfp_training_pipeline.py
```

The console output should look like this:
```
--------------------------------------------------------------------------------------------------------- benchmark: 4 tests ---------------------------------------------------------------------------------------------------------
Name (time in s)                                                            Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_training_duo_modules_e2e[dfp_training_duo_modules_e2e]         14.6904 (1.0)      15.4328 (1.0)      15.0454 (1.0)      0.3456 (1.29)     14.9375 (1.0)      0.6445 (1.33)          2;0  0.0665 (1.0)           5           1
test_dfp_training_duo_stages_e2e[dfp_training_duo_stages_e2e]           16.1813 (1.10)     16.7849 (1.09)     16.4841 (1.10)     0.2677 (1.0)      16.5409 (1.11)     0.4859 (1.0)           2;0  0.0607 (0.91)          5           1
test_dfp_training_azure_stages_e2e[dfp_training_azure_stages_e2e]       27.9413 (1.90)     30.3021 (1.96)     29.3920 (1.95)     0.8975 (3.35)     29.5483 (1.98)     1.0533 (2.17)          2;0  0.0340 (0.51)          5           1
test_dfp_training_azure_modules_e2e[dfp_training_azure_modules_e2e]     29.2428 (1.99)     30.8350 (2.00)     30.1221 (2.00)     0.6535 (2.44)     30.0971 (2.01)     1.0755 (2.21)          2;0  0.0332 (0.50)          5           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Benchmarks Report

Each time you run the benchmarks as above, a comprehensive report for each run will be generated and saved to a JSON file in  `./benchmarks/.benchmarks`. The file name will begin
with `000N` where N is incremented for every run. For example, the report file name for first benchmarks run will look like:
```
0001_f492e1952d5981527d89229e557006a1db992e5f_20230201_230822.json
```

A hook to `pytest-benchmark` was developed to add the following information to the JSON report:

CPU(s) used by Morpheus. For example:
```
"cpu": {
    "python_version": "3.8.15.final.0 (64 bit)",
    "cpuinfo_version": [
        9,
        0,
        0
    ],
    "cpuinfo_version_string": "9.0.0",
    "arch": "X86_64",
    "bits": 64,
    "count": 12,
    "arch_string_raw": "x86_64",
    "vendor_id_raw": "GenuineIntel",
    "brand_raw": "Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz",
    "hz_advertised_friendly": "3.4000 GHz",
    "hz_actual_friendly": "3.4000 GHz",
}
```

Morpheus config for each workflow:
- num_threads
- pipeline_batch_size
- edge_buffer_size
- start_time
- duration
