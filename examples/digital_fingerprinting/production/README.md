# "Production" Digital Fingerprinting Pipeline

This example is designed to show what a full scale, production ready, DFP deployment in Morpheus would look like. It contains all of the necessary components (such as a model store), to allow multiple Morpheus pipelines to communicate at a scale that can handle the workload of an entire company.

Key Differences:
 * Multiple pipelines are specialized to perform either training or inference
 * Requires setting up a model store to allow the training and inference pipelines to communicate
 * Organized into a docker-compose deployment for easy startup
 * Contains a Jupyter notebook service to ease development and debugging
 * Can be deployed to Kubernetes using provided Helm charts
 * Uses many customized stages to maximize performance.

## Build the Morpheus container
This is necessary to get the latest changes needed for DFP. From the root of the Morpheus repo:
```bash
./docker/build_container_release.sh
```

## Buildign and Running via `docker-compose`
### Build
```bash
cd examples/digital_fingerprinting/production
export MORPHEUS_CONTAINER_VERSION="$(git describe --tags --abbrev=0)-runtime"
docker-compose build
```

### Runing the services
#### Jupyter Server
From the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose up jupyter
```

Once the build is complete and the service has started you will be prompted with a message that should look something like:
```
jupyter  |     To access the server, open this file in a browser:
jupyter  |         file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
jupyter  |     Or copy and paste one of these URLs:
jupyter  |         http://localhost:8888/lab?token=<token>
jupyter  |      or http://127.0.0.1:8888/lab?token=<token>
```

Copy and paste the url into a web browser. There are four notebooks included with the DFP example:
* dfp_azure_training.ipynb - Training pipeline for Azure Active Directory data
* dfp_azure_inference.ipynb - Inference pipeline for Azure Active Directory data
* dfp_duo_training.ipynb - Training pipeline for Duo Authentication
* dfp_duo_inference.ipynb - Inference pipeline for Duo Authentication

> **Note:** The token in the url is a one-time use token, and a new one is generated with each invocation.

#### Morpheus Pipeline
By default the `morpheus_pipeline` will run the training pipeline for Duo data, from the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose up morpheus_pipeline
```

If instead you wish to run a different pipeline, from the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose run morpheus_pipeline bash
```

From the prompt within the `morpheus_pipeline` container you can run either the `dfp_azure_pipeline.py` or `dfp_duo_pipeline.py` pipeline scripts.
```bash
python dfp_azure_pipeline.py --help
python dfp_duo_pipeline.py --help
```

Both scripts are capable of running either a training or inference pipeline for their respective data sources. The command line options for both are the same:
| Flag | Type | Description |
| ---- | ---- | ----------- |
| `--train_users` | One of: `all`, `generic`, `individual`, `none` | Indicates whether or not to train per user or a generic model for all users. Selecting `none` runs the inference pipeline. |
| `--skip_user` | TEXT | User IDs to skip. Mutually exclusive with `only_user` |
| `--only_user` | TEXT | Only users specified by this option will be included. Mutually exclusive with `skip_user` |
| `--duration` | TEXT | The duration to run starting from now [default: 60d] |
| `--cache_dir` | TEXT | The location to cache data such as S3 downloads and pre-processed data  [env var: `DFP_CACHE_DIR`; default: `./.cache/dfp`] |
| `--log_level` | One of: `CRITICAL`, `FATAL`, `ERROR`, `WARN`, `WARNING`, `INFO`, `DEBUG` | Specify the logging level to use.  [default: `WARNING`] |
| `--sample_rate_s` | INTEGER | Minimum time step, in milliseconds, between object logs.  [env var: `DFP_SAMPLE_RATE_S`; default: 0] |
| `-f`, `--input_file` | TEXT | List of files to process. Can specify multiple arguments for multiple files. Also accepts glob (*) wildcards and schema prefixes such as `s3://`. For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. See [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files) for list of possible options. |
| `--tracking_uri` | TEXT | The MLflow tracking URI to connect to the tracking backend. [default: `http://localhost:5000`] |
| `--help` | | Show this message and exit. |


#### Optional MLflow Service
Starting either the `morpheus_pipeline` or the `jupyter` service, will start the `mlflow` service in the background.  For debugging purposes it can be helpful to view the logs of the running MLflow service.

From the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose up mlflow
```

## Kubernetes deployment

The Morpheus project also maintains Helm charts and container images for Kubernetes deployment of Morpheus and MLflow (both for serving and for the Triton plugin). These are located in the NVIDIA GPU Cloud (NGC) [public catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/collections/morpheus_).

### MLflow Helm chart

MLflow for this production digital fingerprint use case can be installed from NGC using these same instructions for the [MLflow Triton Plugin from the Morpheus Quick Start Guide](../../../docs/source/morpheus_quickstart_guide.md#install-morpheus-mlflow-triton-plugin). The chart and image can be used for both the Triton plugin and also MLflow server.

### Production DFP Helm chart

The deployment of the [Morpheus SDK Client](../../../docs/source/morpheus_quickstart_guide.md#install-morpheus-sdk-client) is also done _almost_ the same way as what's specified in the Quick Start Guide. However, you would specify command arguments differently for this production DFP use case.

#### Notebooks

```
helm install --set ngc.apiKey="$API_KEY",sdk.args="cd /workspace/examples/digital_fingerprinting/production/morpheus && jupyter-lab --ip='*' --no-browser --allow-root --ServerApp.allow_origin='*'" <sdk-release-name> morpheus-sdk-client/
```

Make note of the Jupyter token by examining the logs of the SDK pod:
```
kubectl logs sdk-cli-<sdk-release-name>
```

You should see something similar to this:

```
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=d16c904468fdf666c5030e18fb82f840e531178bf716e575
     or http://127.0.0.1:8888/lab?token=d16c904468fdf666c5030e18fb82f840e531178bf716e575
```

Open your browser to the reachable address and NodePort exposed by the pod (default value of 30888) and use the generated token to login into the notebook.

#### Unattended

```
helm install --set ngc.apiKey="$API_KEY",sdk.args="cd /workspace/examples/digital_fingerprinting/production/morpheus && ./launch.sh --train_users=generic --duration=1d" <sdk-release-name> morpheus-sdk-client/
```