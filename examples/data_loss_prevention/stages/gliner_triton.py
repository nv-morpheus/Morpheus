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

import typing

import torch
import tritonclient.grpc as tritonclient

if typing.TYPE_CHECKING:
    from gliner import GLiNER


class GliNERTritonInference:

    def __init__(
            self,
            model_source_dir: str,
            server_url: str = "localhost:8001",
            triton_model_name: str = "gliner_bi_encoder",
            gliner_threshold: float = 0.3,
            onnx_path: str = "model.onnx",  # relative to the model_source_dir
            map_location: str = "cuda"):

        # We load the model locally to use its pre/post-processing functions.
        # The actual heavy inference will be done on Triton.
        self._model = None
        self._model_source_dir = model_source_dir
        self._onnx_path = onnx_path
        self._map_location = map_location
        self.triton_model_name = triton_model_name
        self.gliner_threshold = gliner_threshold
        self.labels_embeddings = torch.tensor([])
        self.client = tritonclient.InferenceServerClient(url=server_url)

    @property
    def model(self) -> "GLiNER":
        """
        Return the GLiNER model instance.
        """
        if self._model is None:
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(self._model_source_dir,
                                                 local_files_only=True,
                                                 onnx_path=self._onnx_path,
                                                 map_location=self._map_location)
        return self._model

    def post_process_results(self, logits_tensor, raw_batch, texts) -> list:
        """
        Post-process the results from the ONNX model.
        """
        onnx_results = self.model.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            logits_tensor,
            flat_ner=True,
            threshold=self.gliner_threshold,
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

    def pre_process(self, texts, labels):
        """
        Pre-process the data for the ONNX model.
        """
        # === 1. PRE-PROCESSING ===
        if self.labels_embeddings.numel() == 0:
            self.labels_embeddings = self.model.encode_labels(labels)

        model_input, raw_batch = self.model.prepare_model_inputs(texts, labels, prepare_entities=False)

        # Convert torch tensors to numpy for Triton
        onnx_inputs = {
            "labels_embeddings": self.labels_embeddings.cpu().numpy(),
            "input_ids": model_input["input_ids"].cpu().numpy(),
            "attention_mask": model_input["attention_mask"].cpu().numpy(),
            "words_mask": model_input["words_mask"].cpu().numpy(),
            "text_lengths": model_input["text_lengths"].cpu().numpy(),
            "span_idx": model_input["span_idx"].cpu().numpy(),
            "span_mask": model_input["span_mask"].cpu().numpy(),
        }

        return onnx_inputs, raw_batch

    def process(self, texts: list[str], labels: list[str]):
        """
        Performs full NER pipeline: pre-process, infer, post-process.
        """

        # === 1. PRE-PROCESSING ===
        onnx_inputs, raw_batch = self.pre_process(texts, labels)

        # === 2. TRITON INFERENCE ===

        # Create InferInput objects
        triton_inputs = []

        for name, data in onnx_inputs.items():
            triton_input = tritonclient.InferInput(name, data.shape, tritonclient.np_to_triton_dtype(data.dtype))
            triton_input.set_data_from_numpy(data)
            triton_inputs.append(triton_input)

        # Request output
        triton_outputs = [tritonclient.InferRequestedOutput("output")]

        # Get response
        response = self.client.infer(self.triton_model_name, inputs=triton_inputs, outputs=triton_outputs)
        logits_np = response.as_numpy("output")

        # === 3. POST-PROCESSING ===
        logits = torch.from_numpy(logits_np).to(self.model.device)

        return self.post_process_results(logits, raw_batch, texts)
