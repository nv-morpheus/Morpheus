#!/usr/bin/env python3
"""
Pre-flight + NIM-shaped correctness gate for the MD-enabled Triton server (TRT-28040).

Proves the integration point that BOTH Morpheus and NeMo-Retriever embedding NIMs rely
on: a TensorRT Multi-Device engine, sharded across 2 GPUs by the MD `tensorrt` backend,
returns the same result as the unsharded 1-GPU engine -- transparently, over the stock
Triton gRPC/HTTP API (the exact API a Morpheus TritonInferenceStage or a NIM speaks).

  mlp_sd : KIND_GPU, 1 GPU, unsharded engine            (reference)
  mlp_md : KIND_MODEL, enable_multi_device, GPUs 0+1     (tensor-parallel sharded)

Same seeded input through both -> assert max relative error < THRESH.
This is the framework-agnostic proof: if this passes, Morpheus/NIM "just work" because
they are transparent Triton clients -- no framework code touches the engine.

Run inside the tritonserver container (has tritonclient) or any host with
`pip install tritonclient[grpc] numpy`, against the running MD server.
"""
import argparse
import sys

import numpy as np
import tritonclient.grpc as grpcclient

SHAPE = (2048, 4096)
THRESH = 2e-2  # weight-TP MLP is ~1e-4; loose bound also covers the embedding/NIM case


def infer(client, model, x):
    inp = grpcclient.InferInput("X", list(x.shape), "FP32")
    inp.set_data_from_numpy(x)
    out = grpcclient.InferRequestedOutput("Y")
    res = client.infer(model_name=model, inputs=[inp], outputs=[out])
    return res.as_numpy("Y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--sd-model", default="mlp_sd")
    ap.add_argument("--md-model", default="mlp_md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = grpcclient.InferenceServerClient(url=args.url)
    for m in (args.sd_model, args.md_model):
        if not client.is_model_ready(m):
            print(f"FAIL: model '{m}' not READY on {args.url}")
            return 2

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal(SHAPE, dtype=np.float32)

    y_sd = infer(client, args.sd_model, x)
    y_md = infer(client, args.md_model, x)

    denom = np.maximum(np.abs(y_sd), 1e-6)
    rel = np.abs(y_md - y_sd) / denom
    rel_max, rel_mean = float(rel.max()), float(rel.mean())

    # embedding-NIM view: cosine similarity per row (NeMo-Retriever ranks by cosine)
    a = y_sd.reshape(SHAPE[0], -1)
    b = y_md.reshape(SHAPE[0], -1)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

    print(f"1-GPU (mlp_sd) out: {y_sd.shape}  2-GPU (mlp_md) out: {y_md.shape}")
    print(f"rel_max={rel_max:.3e}  rel_mean={rel_mean:.3e}  cos_min={float(cos.min()):.6f}")
    ok = rel_max < THRESH
    print("PASS" if ok else "FAIL", f"(threshold rel_max < {THRESH})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
