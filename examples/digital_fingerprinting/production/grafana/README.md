<!--
# Copyright (c) 2023, NVIDIA CORPORATION.
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

# Grafana DFP Dashboard Example

This example demonstrates how to use [Grafana](https://grafana.com/grafana/) to visualize the inference results from the [Azure DFP pipeline example](../production/README.md).

## Grafana Configuration

### CSV data source plugin

The [CSV data source plugin](https://grafana.com/grafana/plugins/marcusolsson-csv-datasource/) is installed to Grafana to read the Azure inference results CSV file. This example assumes we are using the CSV file generated from running the Python script for [Azure DFP pipeline example](../production/README.md).

If using the [notebook version](../production/morpheus/notebooks/dfp_azure_inference.ipynb) to run inference, you'll need to update the `url` in [datasources.yaml](./datasources/datasources.yaml) as follows:
```
url: /workspace/notebooks/dfp_detections_azure.csv
```

Please note that the use of the CSV plugin is for demonstration purposes only. Grafana includes support for many data sources more suitable for production deployments. See [here](https://grafana.com/docs/grafana/latest/datasources/) for more information.

### Updates to grafana.ini

The following is added to the default `grafana.ini` to enable local mode for CSV data source plugin. This allows the CSV data source plugin to access files on local file system.

```
[plugin.marcusolsson-csv-datasource]
allow_local_mode = true
```

## Run Azure Production DFP Training and Inference Examples

### Start Morpheus DFP pipeline container

The following steps are taken from [Azure DFP pipeline example](../production/README.md). Run the followng commands to start the Morpheus container:

Build the Morpheus container:

```bash
./docker/build_container_release.sh
```

Build `docker-compose` services:

```
cd examples/digital_fingerprinting/production
export MORPHEUS_CONTAINER_VERSION="$(git describe --tags --abbrev=0)-runtime"
docker compose build
```

Create `bash` shell in `morpheus_pipeline` container:

```bash
docker compose run morpheus_pipeline bash
```

### Run Azure training pipeline

Run the following in the container to train Azure models.

```bash
python dfp_azure_pipeline.py --log_level INFO --train_users generic --start_time "2022-08-01" --input_file="../../../data/dfp/azure-training-data/AZUREAD_2022*.json"
```

### Run Azure inference pipeline:

Run the inference pipeline with `filter_threshold=0.0`. This will disable the filtering of the inference results.

```bash
python dfp_azure_pipeline.py --log_level INFO --train_users none  --start_time "2022-08-30" --input_file="../../../data/dfp/azure-inference-data/*.json" --filter_threshold=0.0 
```

The inference results will be saved to `dfp_detection_azure.csv` in the directory where script was run.

## Run Grafana Docker Image

To start Grafana, run the following command on host in `examples/digital_fingerprinting/production`:

```
docker compose up grafana
```

## View DFP Dashboard

Our Grafana DFP dashboard can now be accessed via web browser at http://localhost:3000/dashboards.

Log in with admin/admin.

Click on `DFP_Dashboard` in the `General` folder. You may need to expand the `General` folder to see the link.

<img src="./img/screenshot.png">

The dashboard has the following visualization panels:

- Time series of absolute `mean_abs_z` across all rows. We can observe the higher `mean_abs_z` scores for `attacktarget@domain.com` as expected.
- Time series of z-loss per feature across all rows. We can observe the higher z-loss scores for the `appincrement` and `logcount` features.
- Bar gauge of maximum `mean_abs_z` per user.
- Bar gauge of maximum z-loss per feature.
- Table view of all rows/columns in CSV file.