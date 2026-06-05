#!/usr/bin/env python3
"""
Morpheus PoC: multi-GPU (TensorRT Multi-Device) inference via an MD-enabled Triton
backend -- with ZERO Morpheus code changes (TRT-28040).

Morpheus runs all inference through `TritonInferenceStage`, a transparent gRPC client
to a Triton server. The TensorRT Multi-Device feature (a single engine sharded across
>=2 GPUs by the MD `tensorrt` backend via NCCL) is entirely server-side, so a Morpheus
pipeline gets multi-GPU TRT inference simply by pointing `model_name` at an MD model.
The ONLY difference between single-GPU and 2-GPU here is `model_name="mlp_sd"` vs
`"mlp_md"` -- no new stage, no device flags (TritonInferenceStage takes only
model_name / server_url / input_mapping / output_mapping).

Pipeline:
  InMemorySourceStage (seeded 2048x4096 DataFrame)
    -> DeserializeStage
    -> BuildInputStage      (custom: DataFrame -> inference tensor "X" [2048,4096])
    -> TritonInferenceStage (model_name = mlp_md | mlp_sd)
    -> InMemorySinkStage    (capture output tensor "Y")

Run it twice (mlp_sd then mlp_md) on the same seeded input and compare -> proves the
2-GPU MD path returns the 1-GPU result through Morpheus.

Usage (inside a Morpheus container, MD Triton server reachable):
  python morpheus_md_pipeline.py --server-url localhost:8001 --model mlp_md
  python morpheus_md_pipeline.py --server-url localhost:8001 --compare   # sd vs md gate
"""
import argparse
import logging
import sys
import typing

import mrc
import numpy as np
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import TensorMemory
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging

import cudf
import cupy as cp

# In the Morpheus 25.06 container, cudf's device->host copy path (used by __repr__)
# hits a numba/cuda-python binding conflict. The pipeline never needs a host copy --
# only Morpheus' stage-construction logging str()s the source DataFrame -- so make the
# repr cheap and host-free. (Cosmetic; does not affect any data movement.)
cudf.DataFrame.__repr__ = lambda self: f"cudf.DataFrame(shape={self.shape})"

ROWS, FEAT = 2048, 4096


class BuildInputStage(SinglePortStage):
    """Turn the incoming DataFrame into the model's inference input tensor "X".

    Mirrors the canonical Morpheus pattern (a SinglePortStage that sets a
    TensorMemory on the ControlMessage); the built-in PreprocessFILStage does the
    same thing in C++ for tabular models. The tensor name "X" matches the Triton
    model input; TritonInferenceStage maps it through automatically.
    """

    @property
    def name(self) -> str:
        return "build-input"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def _to_tensor(self, msg: ControlMessage) -> ControlMessage:
        df = msg.payload().get_data()
        # DLPack: a pure libcudf<->cupy device handoff. cudf's .values/.to_cupy() route
        # through numba, whose CUDA binding is broken in this container's MRC worker
        # threads; DLPack never touches numba.
        x = cp.from_dlpack(df.to_dlpack()).astype(cp.float32, copy=False)   # [2048, 4096]
        msg.tensors(TensorMemory(count=x.shape[0], tensors={"X": x}))
        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._to_tensor))
        builder.make_edge(input_node, node)
        return node


def run_once(model_name: str, server_url: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((ROWS, FEAT), dtype=np.float32)
    df = cudf.DataFrame(data, columns=[f"f{i}" for i in range(FEAT)])

    config = Config()
    config.mode = PipelineModes.OTHER
    config.feature_length = FEAT
    config.pipeline_batch_size = ROWS
    config.model_max_batch_size = ROWS
    config.num_threads = 1

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(DeserializeStage(config, ensure_sliceable_index=False))
    pipe.add_stage(BuildInputStage(config))
    pipe.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=server_url,
            force_convert_inputs=True,
            input_mapping={"X": "X"},
            output_mapping={"Y": "Y"},
        ))
    sink = InMemorySinkStage(config)
    pipe.add_stage(sink)
    pipe.run()

    msgs = sink.get_messages()
    assert msgs, f"no output messages for model {model_name}"
    y = msgs[0].tensors().get_tensor("Y")
    return cp.asnumpy(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="localhost:8001")
    ap.add_argument("--model", default="mlp_md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true",
                    help="run mlp_sd and mlp_md on the same input and assert they match")
    args = ap.parse_args()

    configure_logging(log_level=logging.INFO)

    if not args.compare:
        y = run_once(args.model, args.server_url, args.seed)
        print(f"{args.model}: output {y.shape} mean={y.mean():.5f}")
        return 0

    y_sd = run_once("mlp_sd", args.server_url, args.seed)
    y_md = run_once("mlp_md", args.server_url, args.seed)
    rel = np.abs(y_md - y_sd) / np.maximum(np.abs(y_sd), 1e-6)
    rel_max = float(rel.max())
    print(f"Morpheus mlp_sd vs mlp_md  rel_max={rel_max:.3e}")
    ok = rel_max < 2e-2
    print("PASS" if ok else "FAIL", "(Morpheus drives 2-GPU MD Triton == 1-GPU, no code change)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
