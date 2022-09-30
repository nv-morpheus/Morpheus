# Triton Model Repository

This directory contains the necessary directory structure and configuration files in order to run the Morpheus Models in Triton Inference Server.

Each directory in the Triton Model Repo contains a single model and configuration file. The model file is stored in a directory indicating the version number (by default this is `1`). The model file itself is a symlink to a specific model file elsewhere in the repo.

For example, the Triton model `sid-minibert-onnx` can be found in the `triton-model-repo` directory with the following layout:

```
triton-model-repo/
   sid-minibert-onnx/
      1/
         model.onnx -> ../../../sid-models/sid-bert-20211021.onnx
      config.pbtxt
```

## Symbolic Links

Sym links are used to minimize changes to the `config.pbtxt` files while still allowing for new models to be added at a future date. Without symlinks, each `config.pbtxt` would need to update the `default_model_filename:` option each time the model was changed.

The downside of using symlinks is that the entire morpheus model repo must be volume mounted when launching Triton. See the next section for information on how to correctly mount this repo and select which models should be loaded.

## Launching Triton

To launch Triton with one of the models in `triton-model-repo`, this entire repo must be volume mounted into the container. Once the entire repository is mounted, the Triton options: `--model-repository` and `--load-model` can be selectively used to choose which models to load. The following are several examples on launching Triton with different models and different setups:

### Load `sid-minibert-onnx` Model with Default Triton Image

```bash
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD:/models --name tritonserver nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model sid-minibert-onnx
```

### Load `abp-nvsmi-xgb` Model with FIL Backend Triton

```bash
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD:/models --name tritonserver triton_fil tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model abp-nvsmi-xgb
```

### Load `sid-minibert-trt` Model with Default Triton Image from Morpheus Repo

To load a TensorRT model, it first must be compiled with the `morpheus tools onnx-to-trt` utility (See `triton-model-repo/sid-minibert-trt/1/README.md` for more info):

```bash
cd models/triton-model-repo/sid-minibert-trt/1
morpheus tools onnx-to-trt --input_model ../../sid-minibert-onnx/1/sid-minibert.onnx --output_model ./sid-minibert-trt_b1-8_b1-16_b1-32.engine --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 256 --max_workspace_size 16000
```

Then launch Triton:

```bash
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/models:/models --name tritonserver nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model sid-minibert-trt
```
