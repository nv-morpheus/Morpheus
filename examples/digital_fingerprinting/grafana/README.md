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

### Run Azure training pipeline

Follow the instructions in [Azure DFP pipeline example](../production/README.md) to run training pipeline using `dfp_azure_pipeline.py`.

### Run Azure inference pipeline:

Run the inference pipeline with `filter_threshold=0.0`. This will disable the filtering of the inference results.
```
python dfp_azure_pipeline.py --train_users none  --start_time "2022-08-30" --input_file="../../../data/dfp/azure-inference-data/*.json" --filter_threshold=0.0 
```

The inference results will be saved to `dfp_detection_azure.csv` in the directory where script was run.

## Run Grafana Docker Image

Run the following commands to start Grafana:

```
cd examples/digital_fingerprinting
```

```
DFP_HOME=${pwd}
```

```
docker run \
-p 3000:3000 \
-v $DFP_HOME/grafana/config/grafana.ini:/etc/grafana/grafana.ini \
-v $DFP_HOME/grafana/config/dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml \
-v $DFP_HOME/grafana/dashboards/:/var/lib/grafana/dashboards/ \
-v $DFP_HOME/grafana/datasources/:/etc/grafana/provisioning/datasources/ \
-v $DFP_HOME/production/morpheus:/workspace \
-e "GF_INSTALL_PLUGINS=marcusolsson-csv-datasource" \
--rm \
grafana/grafana:10.0.0
```

## View DFP Dashboard

Our Grafana DFP dashboard can now be accessed via web browser at http://localhost:3000/dashboards.

Log in with admin/admin.

Click on `DFP_Dashboard` in the `General` folder.

<img src="./img/screenshot.png">

The dashboard has the following visualization panels:

- Time series of absolute `mean_abs_z` across all rows. We can observe the higher `mean_abs_z` scores for `attacktarget@domain.com` as expected.
- Time series of z-loss per feature across all rows. We can bserve the higher z-loss scores for the `appincrement` and `logcount` features.
- Bar gauge of maximum `mean_abs_z` per user.
- Bar gauge of maximum z-loss per feature.
- Table view of all rows/columns in CSV file.