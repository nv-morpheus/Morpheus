### Launching Triton

Pull the Docker image for Triton:
```bash
docker pull nvcr.io/nvidia/tritonserver:23.06-py3
```

From the Morpheus repo root directory, run the following to launch Triton and load the `all-MiniLM-L6-v2` model:
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model all-MiniLM-L6-v2
```

This will launch Triton and only load the `all-MiniLM-L6-v2` model. Once Triton has loaded the model, the following will be displayed:
```
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| all-MiniLM-L6-v2 | 1       | READY  |
+------------------+---------+--------+
```

## Running the Pipeline

```bash
python examples/llm/main.py vdb_upload pipeline
``````

> **Note**: This pipeline will, by default, run continuously repeatedly polling the configured RSS sources. To run for a fixed number of iterations, add the `--stop_after=N` flag.
