# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import logging
import os
import typing
from functools import partial

import numpy as np
import torch
import tritonclient.grpc as tritonclient

if typing.TYPE_CHECKING:
    from gliner import GLiNER

logger = logging.getLogger(f"morpheus.{__name__}")


class GliNERTritonInference:
    """
    Triton client for GLiNER model inference.
    This class uses the local model for pre and post-processing, while the actual inference is done on the Triton
    hosted instance of the model.

    Parameters
    ----------
    model_source_dir : str
        Path to the directory containing the GLiNER model files. Used for pre and post-processing.
    labels : list[str]
        List of entity labels to detect, this should match the named patterns used in the RegexProcessor stage.
        Ignored if the `label_embedding.pt` file is present in the `model_source_dir`.
    onnx_path : str, default = "model.onnx"
        Path to the ONNX model file, relative to the `model_source_dir`. Default is "model.onnx".
    server_url : str, default = "localhost:8001"
        URL of the Triton inference server.
    triton_model_name : str, default = "gliner-bi-encoder-onnx"
        Name of the Triton model to use for inference.
    gliner_threshold: float, default = 0.3
        Minimum confidence score to report a finding
    map_location : str, default = "cuda"
        Device to load the model on. Default is "cuda". Can also be set to "cpu".
        Note this only affects the local model used for pre/post-processing, and has no impact on the inference which
        is done in Triton.
    fallback_model_name: str, default="gretelai/gretel-gliner-bi-small-v1.0"
        Name of the fallback model to use if the local model is not present.
    """

    def __init__(self,
                 model_source_dir: str,
                 labels: list[str],
                 onnx_path: str = "model.onnx",
                 server_url: str = "localhost:8001",
                 triton_model_name: str = "gliner-bi-encoder-onnx",
                 gliner_threshold: float = 0.3,
                 map_location: str = "cuda",
                 fallback_model_name: str = "gretelai/gretel-gliner-bi-small-v1.0"):

        # We load the model locally to use its pre/post-processing functions.
        # The actual heavy inference will be done on Triton.
        self._model = None
        self._model_source_dir = model_source_dir
        self._onnx_path = onnx_path
        self._map_location = map_location
        self._triton_model_name = triton_model_name
        self._gliner_threshold = gliner_threshold
        self._labels_embeddings: np.ndarray | None = None
        self._labels: list[str] | None = None
        self._user_labels: list[str] = labels
        self._labels_file = os.path.join(model_source_dir, "label_embedding.pt")
        self._fallback_model_name = fallback_model_name

        self._client = tritonclient.InferenceServerClient(url=server_url)

    @property
    def model(self) -> "GLiNER":
        """
        Return the GLiNER model instance.
        """
        if self._model is None:
            from gliner import GLiNER

            if os.path.exists(self._model_source_dir) and os.path.exists(
                    os.path.join(self._model_source_dir, self._onnx_path)):
                model_kwargs = {
                    "pretrained_model_name_or_path": self._model_source_dir,
                    "local_files_only": True,
                    "map_location": self._map_location,
                    "onnx_path": self._onnx_path,
                    "load_onnx_model": True
                }
            else:
                model_kwargs = {
                    "pretrained_model_name_or_path": self._fallback_model_name,
                    "local_files_only": False,
                    "map_location": self._map_location
                }

            self._model = GLiNER.from_pretrained(**model_kwargs)
        return self._model

    def _load_label_data(self):
        if os.path.exists(self._labels_file):
            label_data = torch.load(self._labels_file)

            labels_embeddings = label_data['embeddings']
            self._labels = label_data['labels']
        else:
            self._labels = self._user_labels
            labels_embeddings = self.model.encode_labels(self._labels)

        self._labels_embeddings = labels_embeddings.cpu().numpy()

    @property
    def labels_embeddings(self) -> np.ndarray:
        """
        Return the labels embeddings tensor.
        If not loaded, it will load from the specified file.
        """
        if self._labels_embeddings is None:
            self._load_label_data()

        return self._labels_embeddings

    @property
    def labels(self) -> list[str]:
        """
        Return the list of labels.
        If not loaded, it will load from the specified file.
        """
        if self._labels is None:
            self._load_label_data()

        return self._labels

    def pre_process(self, texts):
        """
        Pre-process the data for the ONNX model.
        """
        # === 1. PRE-PROCESSING ===
        model_input, raw_batch = self.model.prepare_model_inputs(texts, self.labels, prepare_entities=False)

        # Convert torch tensors to numpy for Triton
        onnx_inputs = [
            ("labels_embeddings", self.labels_embeddings),
            ("input_ids", model_input["input_ids"].cpu().numpy()),
            ("attention_mask", model_input["attention_mask"].cpu().numpy()),
            ("words_mask", model_input["words_mask"].cpu().numpy()),
            ("text_lengths", model_input["text_lengths"].cpu().numpy()),
            ("span_idx", model_input["span_idx"].cpu().numpy()),
            ("span_mask", model_input["span_mask"].cpu().numpy()),
        ]

        return onnx_inputs, raw_batch

    def _process_inference_results(self, logits_tensor, raw_batch, texts) -> list:
        """
        Post-process the results from the ONNX model.
        """
        onnx_results = self.model.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            logits_tensor,
            flat_ner=True,
            threshold=self._gliner_threshold,
            multi_label=False,
        )

        # Process results to match the expected format
        all_entities = []
        for i, output in enumerate(onnx_results):
            start_token_idx_to_text_idx = raw_batch["all_start_token_idx_to_text_idx"][i]
            end_token_idx_to_text_idx = raw_batch["all_end_token_idx_to_text_idx"][i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                entities.append({
                    "start": start_token_idx_to_text_idx[start_token_idx],
                    "end": end_token_idx_to_text_idx[end_token_idx],
                    "text": texts[i][start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score,
                })
            all_entities.append(entities)
        return all_entities

    def _infer_callback(self,
                        raw_batch: dict,
                        texts: list[str],
                        cb,
                        result: tritonclient.InferResult,
                        error: tritonclient.InferenceServerException | None = None):
        if (error is not None):
            logger.error("Triton inference failed: %s", error)
            raise error

        logits_np = result.as_numpy("output")

        # === 3. POST-PROCESSING ===
        logits = torch.from_numpy(logits_np).to(self.model.device)
        entities = self._process_inference_results(logits, raw_batch, texts)
        cb(entities=entities)

    def process(self, texts: list[str], callback):
        """
        Performs full NER pipeline: pre-process, infer, post-process.
        """

        # === 1. PRE-PROCESSING ===
        onnx_inputs, raw_batch = self.pre_process(texts)

        # === 2. TRITON INFERENCE ===

        # Create InferInput objects
        triton_inputs = []

        for (name, data) in onnx_inputs:
            triton_input = tritonclient.InferInput(name, data.shape, tritonclient.np_to_triton_dtype(data.dtype))
            triton_input.set_data_from_numpy(data)
            triton_inputs.append(triton_input)

        # Request output
        triton_outputs = [tritonclient.InferRequestedOutput("output")]

        # Get response
        self._client.async_infer(self._triton_model_name,
                                 inputs=triton_inputs,
                                 outputs=triton_outputs,
                                 callback=partial(self._infer_callback, raw_batch, texts, callback))
