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
docker pull nvcr.io/nvidia/tritonserver:22.08-py3
```

##### Start Triton Inference Server container
```
cd ${MORPHEUS_ROOT}/models

docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models/triton-model-repo --model-control-mode=explicit --load-model sid-minibert-onnx --load-model abp-nvsmi-xgb --load-model phishing-bert-onnx
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

### Set up Morpheus Dev Container

If you don't already have the Morpheus Dev container, run the following to build it:
```
./docker/build_container_dev.sh
```

Now run the container:
```
./docker/run_container_dev.sh
```

Note that Morpheus containers are tagged by date. By default, `run_container_dev.sh` will try to use current date as tag. Therefore, if you are trying to run a container that was not built on the current date, you must set the `DOCKER_IMAGE_TAG` environment variable. For example,
```
DOCKER_IMAGE_TAG=dev-221003 ./docker/run_container_dev.sh
```

In the `/workspace` directory of the container, run the following to compile Morpheus:
```
./scripts/compile.sh
```

Now install Morpheus:
```
pip install -e /workspace
```

Fetch input data for benchmarks:
```
./scripts/fetch_data.py fetch validation
```


### Run E2E Benchmarks

Benchmarks are run using `pytest-benchmark`. By default, there are five rounds of measurement. For each round, there will be one iteration of each workflow. Measurements are taken for each round. Final results such as `min`, `max` and `mean` times will be based on these measurements.

To provide your own calibration or use other `pytest-benchmark` features with these workflows, please refer to their [documentation](https://pytest-benchmark.readthedocs.io/en/latest/).

Morpheus configurations for each workflow are managed using `e2e_test_configs.json`. For example, this is the Morpheus configuration for  `sid_nlp`:
```
"test_sid_nlp_e2e": {
    "file_path": "../../models/datasets/validation-data/sid-validation-data.csv",
    "repeat": 10,
    "num_threads": 8,
    "pipeline_batch_size": 1024,
    "model_max_batch_size": 64,
    "feature_length": 256,
    "edge_buffer_size": 4
},
...
```

Benchmarks for an individual workflow can be run using the following:

```
cd tests/benchmarks

pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_pipelines.py::<test-workflow>
```
The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

The `--benchmark-warmup` and `--benchmark-warmup-iterations` options are used to run the workflow(s) once before starting measurements. This is because the models deployed to Triton are configured to convert from ONNX to TensorRT on first use. Since the conversion can take a considerable amount of time, we don't want to include it in the measurements.

`<test-workflow>` is the name of the test to run benchmarks on. This can be one of the following:
- `test_sid_nlp_e2e`
- `test_abp_fil_e2e`
- `test_phishing_nlp_e2e`
- `test_cloudtrail_ae_e2e`

For example, to run E2E benchmarks on the SID NLP workflow:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_pipelines.py::test_sid_nlp_e2e
```

To run E2E benchmarks on all workflows:
```
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_pipelines.py
```

The console output should look like this:
```
--------------------------------------------------------------------------------- benchmark: 4 tests --------------------------------------------------------------------------------
Name (time in s)              Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_sid_nlp_e2e           1.8907 (1.0)      1.9817 (1.0)      1.9400 (1.0)      0.0325 (2.12)     1.9438 (1.0)      0.0297 (1.21)          2;0  0.5155 (1.0)           5           1
test_cloudtrail_ae_e2e     3.3403 (1.77)     3.3769 (1.70)     3.3626 (1.73)     0.0153 (1.0)      3.3668 (1.73)     0.0245 (1.0)           1;0  0.2974 (0.58)          5           1
test_abp_fil_e2e           5.1271 (2.71)     5.3044 (2.68)     5.2083 (2.68)     0.0856 (5.59)     5.1862 (2.67)     0.1653 (6.75)          1;0  0.1920 (0.37)          5           1
test_phishing_nlp_e2e      5.6629 (3.00)     6.0987 (3.08)     5.8835 (3.03)     0.1697 (11.08)    5.8988 (3.03)     0.2584 (10.55)         2;0  0.1700 (0.33)          5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Benchmarks Report

Each time you run the benchmarks as above, a comprehensive report for each run will be generated and saved to a JSON file in  `./tests/benchmarks/.benchmarks`. The file name will begin
with `000N` where N is incremented for every run. For example, the report file name for first benchmarks run will look like:
```
0001_dacccac5198c7eeddc477794bc278028e739c2cd_20220929_182030.json
```

A hook to `pytest-benchmark` was developed to add the following information to the JSON report:

GPU(s) used by Morpheus. For example:
```
"gpu_0": {
    "id": 0,
    "name": "Quadro RTX 8000",
    "load": "0.0%",
    "free_memory": "42444.0MB",
    "used_memory": "6156.0MB",
    "temperature": "61.0 C",
    "uuid": "GPU-dc32de82-bdaa-2d05-2abe-260a847e1989"
}
```

Morpheus config for each workflow:
- num_threads
- pipeline_batch_size
- model_max_batch_size
- feature_length
- edge_buffer_size

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
