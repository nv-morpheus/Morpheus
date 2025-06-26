# Copyright (c) 2025, NVIDIA CORPORATION.
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

import os

import onnxruntime as ort
import torch
from gliner import GLiNER


def convert_bi_encoder_to_onnx(model, onnx_path, example_texts, example_labels, opset_version=14):
    """Convert a bi-encoder GLiNER model to ONNX using pre-encoded labels."""

    print("Pre-encoding labels for bi-encoder model...")

    # Pre-encode labels
    labels_embeddings = model.encode_labels(example_labels)
    print(f"Labels embeddings shape: {labels_embeddings.shape}")

    # Prepare model inputs
    model_input, _ = model.prepare_model_inputs(example_texts, example_labels, prepare_entities=False)

    # Create wrapper model that uses pre-encoded labels
    class BiEncoderONNXWrapper(torch.nn.Module):

        def __init__(self, gliner_model):
            super().__init__()
            self.model = gliner_model.model

        def forward(
            self,
            labels_embeddings,
            input_ids,
            attention_mask,
            words_mask,
            text_lengths,
            span_idx,
            span_mask,
        ):
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "words_mask": words_mask,
                "text_lengths": text_lengths,
                "span_idx": span_idx,
                "span_mask": span_mask,
                "labels_embeddings": labels_embeddings,
            }

            # Remove None values
            model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

            return self.model(**model_inputs)[0]

    # Create wrapper
    wrapper_model = BiEncoderONNXWrapper(model)
    wrapper_model.eval()

    # Prepare input tensors
    span_idx = model_input["span_idx"]
    span_mask = model_input["span_mask"]

    input_tensors = [
        labels_embeddings,
        model_input["input_ids"],
        model_input["attention_mask"],
        model_input["words_mask"],
        model_input["text_lengths"],
        span_idx if span_idx is not None else torch.empty(0),
        span_mask if span_mask is not None else torch.empty(0),
    ]

    input_names = [
        "labels_embeddings",
        "input_ids",
        "attention_mask",
        "words_mask",
        "text_lengths",
        "span_idx",
        "span_mask",
    ]

    # Define dynamic axes
    dynamic_axes = {
        "labels_embeddings": {
            0: "num_labels"
        },
        "input_ids": {
            0: "batch_size", 1: "seq_length"
        },
        "attention_mask": {
            0: "batch_size", 1: "seq_length"
        },
        "words_mask": {
            0: "batch_size", 1: "seq_length"
        },
        "text_lengths": {
            0: "batch_size"
        },
        "span_idx": {
            0: "batch_size", 1: "num_spans"
        },
        "span_mask": {
            0: "batch_size", 1: "num_spans"
        },
        "output": {
            0: "batch_size"
        },
    }

    # Create output directory
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Export the model to ONNX
    torch.onnx.export(
        wrapper_model,
        tuple(input_tensors),
        onnx_path,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    # Save the pre-encoded labels for later use
    labels_path = os.path.join(os.path.dirname(onnx_path), "example_labels_embeddings.pt")
    torch.save({"labels": example_labels, "embeddings": labels_embeddings}, labels_path)

    # Save model and labels to the same directory
    print(f"Saved example labels embeddings to: {labels_path}")
    print("ONNX export completed for bi-encoder model")


def test_onnx_model(onnx_model_dir, test_texts, test_labels):
    """Test the converted ONNX model."""

    print("Testing ONNX model...")

    onnx_path = os.path.join(onnx_model_dir, "model.onnx")
    labels_path = os.path.join(onnx_model_dir, "example_labels_embeddings.pt")

    # Load ONNX session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    ort_session = ort.InferenceSession(onnx_path, session_options)

    # Load the GLiNER model from the saved directory for preprocessing
    gliner_model = GLiNER.from_pretrained(onnx_model_dir)

    # Load pre-encoded labels
    labels_data = torch.load(labels_path)
    labels_embeddings = labels_data["embeddings"]

    print("\n=== ONNX Model Results ===")

    # Prepare inputs using the loaded model
    model_input, raw_batch = gliner_model.prepare_model_inputs(test_texts, test_labels, prepare_entities=False)

    # Prepare ONNX inputs
    onnx_inputs = {
        "labels_embeddings": labels_embeddings.cpu().numpy(),
        "input_ids": model_input["input_ids"].cpu().numpy(),
        "attention_mask": model_input["attention_mask"].cpu().numpy(),
        "words_mask": model_input["words_mask"].cpu().numpy(),
        "text_lengths": model_input["text_lengths"].cpu().numpy(),
    }

    if model_input["span_idx"] is not None:
        onnx_inputs["span_idx"] = model_input["span_idx"].cpu().numpy()
    if model_input["span_mask"] is not None:
        onnx_inputs["span_mask"] = model_input["span_mask"].cpu().numpy()

    # Run ONNX inference
    onnx_outputs = ort_session.run(None, onnx_inputs)
    logits = torch.from_numpy(onnx_outputs[0])

    # Decode results
    onnx_results = gliner_model.decoder.decode(
        raw_batch["tokens"],
        raw_batch["id_to_classes"],
        logits,
        flat_ner=True,
        threshold=0.5,
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
                "text": test_texts[i][start_text_idx:end_text_idx],
                "label": ent_type,
                "score": ent_score,
            })
        all_entities.append(entities)

    for i, (text, entities) in enumerate(zip(test_texts, all_entities)):
        print(f"\nText {i + 1}: {text}")
        if entities:
            for entity in entities:
                print(f"  - {entity['text']} ({entity['label']}) - "
                      f"Score: {entity['score']:.3f}")
        else:
            print("  No entities found")

    # Summary
    print("\n=== Summary ===")
    total_onnx = sum(len(entities) for entities in all_entities)
    print(f"ONNX model found {total_onnx} entities")


def convert_gliner_to_onnx(model_name_or_path, onnx_output_dir, example_texts, example_labels):
    """Convert a GLiNER bi-encoder model to ONNX format."""

    print(f"Loading GLiNER model from: {model_name_or_path}")

    # Load the pre-trained model
    model = GLiNER.from_pretrained(model_name_or_path)

    # Create output directory
    os.makedirs(onnx_output_dir, exist_ok=True)

    # Save model config and tokenizer locally
    model.save_pretrained(onnx_output_dir)

    print("Converting bi-encoder model to ONNX...")

    # Convert to ONNX
    onnx_model_path = os.path.join(onnx_output_dir, "model.onnx")

    convert_bi_encoder_to_onnx(model, onnx_model_path, example_texts, example_labels)

    return model


def main():
    """Main function to convert and test GLiNER bi-encoder model."""

    # Configuration
    model_name = "gretelai/gretel-gliner-bi-small-v1.0"
    output_dir = "gliner_bi_encoder"

    # Sample data for testing
    example_texts = [
        "Apple Inc. is headquartered in Cupertino, California.",
        "The meeting will be held in New York on December 15th, 2023.",
        """ Purchase Order
        ----------------
        Date: 10/05/2023
        ----------------
        Customer Name: CID-982305
        Billing Address: 1234 Oak Street, Suite 400, Springfield, IL, 62704
        Phone: (312) 555-7890 (555-876-5432)
        Email: janedoe@company.com"""
    ]
    # Define the labels for PII/PHI entities
    example_labels = [
        "medical_record_number",
        "date_of_birth",
        "ssn",
        "date",
        "first_name",
        "email",
        "last_name",
        "customer_id",
        "employee_id",
        "name",
        "street_address",
        "phone_number",
        "ipv4",
        "credit_card_number",
        "license_plate",
        "address",
        "user_name",
        "device_identifier",
        "bank_routing_number",
        "date_time",
        "company_name",
        "unique_identifier",
        "biometric_identifier",
        "account_number",
        "city",
        "certificate_license_number",
        "time",
        "postcode",
        "vehicle_identifier",
        "coordinate",
        "country",
        "api_key",
        "ipv6",
        "password",
        "health_plan_beneficiary_number",
        "national_id",
        "tax_id",
        "url",
        "state",
        "swift_bic",
        "cvv",
        "pin"
    ]

    try:
        # Convert model to ONNX
        _ = convert_gliner_to_onnx(
            model_name_or_path=model_name,
            onnx_output_dir=output_dir,
            example_texts=example_texts,
            example_labels=example_labels,
        )

        # Test the ONNX model
        test_onnx_model(
            onnx_model_dir=output_dir,
            test_texts=example_texts,
            test_labels=example_labels,
        )

        print("\n ONNX conversion completed successfully!")
        print(f"ONNX model saved at: {output_dir}")

    except Exception as e:
        print(f" Error during conversion: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
