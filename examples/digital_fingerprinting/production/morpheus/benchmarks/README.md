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

Morpheus pipeline configurations for each workflow are managed using [pipelines_conf.json](./pipelines_conf.json). For example, this is the Morpheus configuration for  `duo_training_modules`:
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

In addition to the Morpheus pipeline settings, we also have a configuration file called [modules_conf.json](./modules_conf.json) that is specific to modules. When using MRC SegmentModule, pipelines need this configuration file. Additional information is included in the [Morpheus Pipeline with Modules](../../../../../docs/source/developer_guide/guides/6_digital_fingerprinting_reference.md#morpheus-pipeline-with-modules)

Benchmarks for an individual workflow can be run using the following:

```

pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave bench_e2e_dfp_training_pipeline.py::<test-workflow>
```
The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

The `--benchmark-warmup` and `--benchmark-warmup-iterations` options are used to run the workflow(s) once before starting measurements. This is because the models deployed to Triton are configured to convert from ONNX to TensorRT on first use. Since the conversion can take a considerable amount of time, we don't want to include it in the measurements.

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
test_dfp_training_duo_modules_e2e[dfp_training_duo_modules_e2e]         13.8403 (1.0)      14.6742 (1.0)      14.3421 (1.0)      0.3276 (1.37)     14.3941 (1.0)      0.4632 (1.81)          2;0  0.0697 (1.0)           5           1
test_dfp_training_duo_stages_e2e[dfp_training_duo_stages_e2e]           14.4521 (1.04)     15.1115 (1.03)     14.8144 (1.03)     0.2397 (1.0)      14.8046 (1.03)     0.2562 (1.0)           2;0  0.0675 (0.97)          5           1
test_dfp_training_azure_modules_e2e[dfp_training_azure_modules_e2e]     26.1414 (1.89)     29.1618 (1.99)     27.5113 (1.92)     1.1336 (4.73)     27.3633 (1.90)     1.5112 (5.90)          2;0  0.0363 (0.52)          5           1
test_dfp_training_azure_stages_e2e[dfp_training_azure_stages_e2e]       27.8076 (2.01)     28.6514 (1.95)     28.2133 (1.97)     0.3722 (1.55)     28.3723 (1.97)     0.6270 (2.45)          3;0  0.0354 (0.51)          5           1
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
