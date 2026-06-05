<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Multi-GPU inference with a TensorRT Multi-Device Triton backend

This example shows a Morpheus pipeline running a **single TensorRT engine sharded across
multiple GPUs** — with **no Morpheus code change**. Morpheus runs all inference through
[`TritonInferenceStage`](../../python/morpheus/morpheus/stages/inference/triton_inference_stage.py),
a transparent gRPC client. TensorRT **Multi-Device** (MD) — a single engine split across ≥2 GPUs
via NCCL, GA in TensorRT 11 — is implemented in the Triton `tensorrt` backend and is fully
server-side. A Morpheus pipeline gets multi-GPU TRT inference simply by pointing `model_name` at
an MD-enabled model.

> The **only** difference between single-GPU and multi-GPU here is `model_name="mlp_sd"` vs
> `model_name="mlp_md"`. `TritonInferenceStage` takes no device arguments.

## Model repository

Two batched models (`max_batch_size = 2048`, input `X` / output `Y` = 4096-d) built from the
same tensor-parallel MLP, served by the **MD-enabled** Triton `tensorrt` backend:

- `model_repo/mlp_sd` — single-GPU baseline (`KIND_GPU`, GPU 0), full engine `1/model.plan`.
- `model_repo/mlp_md` — the same network sharded across GPUs 0+1. The only additions vs an
  ordinary TRT model:

  ```
  instance_group [ { kind: KIND_MODEL count: 1 } ]
  parameters [
    { key: "enable_multi_device"           value: { string_value: "true" } },
    { key: "multi_device_gpus"             value: { string_value: "0,1" } },
    { key: "multi_device_per_rank_engines" value: { string_value: "true" } }
  ]
  ```
  Per-rank weight-shard engines live as `1/model.plan.rank0` / `1/model.plan.rank1`.

`build_tp_engines_batched.cpp` builds the plans (a Megatron MLP: column-parallel `W1`,
row-parallel `W2` + `AllReduce`, dynamic batch). Build it against TensorRT ≥ 11 + NCCL:

```bash
nvcc -std=c++17 -w build_tp_engines_batched.cpp -o build_tp_engines_batched \
     -I$TRT/include -L$TRT/lib -lnvinfer
LD_LIBRARY_PATH=$TRT/lib ./build_tp_engines_batched 2 model_repo_plans   # world=2
```

Start a Triton server whose `tensorrt` backend was built with
`-DTRITON_ENABLE_TENSORRT_MULTI_DEVICE=ON` (TensorRT ≥ 11, NCCL) and run with two GPUs
(`--gpus '"device=0,1"'`). See the Triton `tensorrt_backend` `docs/multi_device.md`.

## Run

```bash
python run.py --server-url localhost:8001 --model mlp_sd   # single-GPU
python run.py --server-url localhost:8001 --model mlp_md   # 2-GPU MD, identical pipeline
python run.py --server-url localhost:8001 --compare        # assert mlp_sd == mlp_md
```

`triton_md_compare.py` is a minimal raw-gRPC equivalent of the inference call the stage issues —
useful to validate the server independently of Morpheus.

## Pipeline

```
InMemorySourceStage  ->  DeserializeStage  ->  BuildInputStage  ->  TritonInferenceStage  ->  InMemorySinkStage
                                              (DataFrame -> "X")   (model_name = mlp_md|mlp_sd)
```

`BuildInputStage` is the canonical Morpheus pattern (a `SinglePortStage` that sets a
`TensorMemory` on the `ControlMessage`); the built-in `PreprocessFILStage` does the same in C++
for tabular models.

## Validation

Validated on 8× B200 (NVLink), Morpheus `25.06-runtime`, against an MD `tensorrt` backend built
on TensorRT 11.1 + NCCL and `tritonserver:25.06-py3` (CUDA 12.9):

- **Morpheus end-to-end** (`run.py --compare`): `mlp_sd` (1 GPU) vs `mlp_md` (2 GPU),
  same seeded input → **`rel_max = 4.78e-3`** → PASS. The pipeline is byte-for-byte identical
  between the two runs except `model_name`.
- **Raw-gRPC gate** (`triton_md_compare.py`): `rel_max = 3.56e-3`, `cos_min = 0.999999`.
- Both GPUs hold the model (rank 1 lives entirely on GPU 1); the server logs
  `TensorRT Multi-Device ready for 'mlp_md_0_0': 2 ranks`.

Latency is reported by `MonitorStage` for information only; a small MLP is overhead-bound and no
speedup is claimed at this scale — the point is *transparent multi-GPU TRT serving* for models
too large for one GPU.
