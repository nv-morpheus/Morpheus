import torch
import tritonclient.http as httpclient
from gliner import GLiNER


class GliNERTritonInference:

    def __init__(self,
                 model_source_dir: str,
                 triton_model_name: str = "gliner_bi_encoder",
                 gliner_threshold: float = 0.3):

        # We load the model locally to use its pre/post-processing functions.
        # The actual heavy inference will be done on Triton.
        self.gliner_model = GLiNER.from_pretrained(model_source_dir,
                                                   local_files_only=True,
                                                   onnx_path="model.onnx",
                                                   map_location="cuda")
        self.triton_model_name = triton_model_name
        self.gliner_threshold = gliner_threshold
        self.labels_embeddings = torch.tensor([])

    def post_process_results(self, logits_tensor, raw_batch, texts) -> list:
        """
        Post-process the results from the ONNX model.
        """
        onnx_results = self.gliner_model.decoder.decode(
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
            self.labels_embeddings = self.gliner_model.encode_labels(labels)

        model_input, raw_batch = self.gliner_model.prepare_model_inputs(texts, labels, prepare_entities=False)

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
        client = httpclient.InferenceServerClient(url="localhost:8000")

        # Create InferInput objects
        triton_inputs = [
            httpclient.InferInput(name, data.shape, httpclient.np_to_triton_dtype(data.dtype))
            for name, data in onnx_inputs.items()
        ]

        # Set data for each input
        for i, name in enumerate(onnx_inputs.keys()):
            triton_inputs[i].set_data_from_numpy(onnx_inputs[name])

        # Request output
        triton_outputs = [httpclient.InferRequestedOutput("output")]

        # Get response
        response = client.infer(self.triton_model_name, inputs=triton_inputs, outputs=triton_outputs)
        logits_np = response.as_numpy("output")

        # === 3. POST-PROCESSING ===
        logits = torch.from_numpy(logits_np)

        return self.post_process_results(logits, raw_batch, texts)
