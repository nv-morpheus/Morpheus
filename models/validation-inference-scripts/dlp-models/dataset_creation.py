# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
from ast import literal_eval

import pandas as pd
from datasets import load_dataset

# random seed for reproducibility
SEED = 42  # type: ignore


def fix_gretel_masks(row):
    """Fix Gretel dataset mask format to match standard format."""
    list_of_masks = []
    for m in row.privacy_mask:
        label = m["types"][0]
        value = m["entity"]
        matches = list(re.finditer(re.escape(value), row.source_text))
        if len(matches) > 1:
            print("WARNING: found more than one of the entity!")
        start, end = matches[0].start(), matches[0].end()

        list_of_masks.append({
            "label": label,
            "start": start,
            "end": end,
            "value": value,
        })

    return list_of_masks


def process_gretel_dataset(dataset, num_samples):
    """Process Gretel dataset to standard format."""
    df = dataset.to_pandas()
    df = df[["text", "entities"]]
    df.columns = ["source_text", "privacy_mask"]

    if num_samples:
        df = df.sample(n=num_samples, random_state=SEED).reset_index(drop=True)

    df["privacy_mask"] = df["privacy_mask"].apply(literal_eval)
    df["privacy_mask"] = df.apply(fix_gretel_masks, axis=1)
    df["source"] = "gretel"

    return df


def load_and_process_datasets(dataset_names, num_samples=None):
    """Load and process specified datasets."""
    available_datasets = {
        "gretel": "gretelai/gretel-pii-masking-en-v1",
    }

    print(f"Loading datasets: {dataset_names}")

    dataframes = []
    for name in dataset_names:
        if name not in available_datasets:
            raise ValueError(f"Unknown dataset: {name}")

        dataset = load_dataset(available_datasets[name], split="validation")

        if name == "gretel":
            df = process_gretel_dataset(dataset, num_samples)
            dataframes.append(df)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    # Combine dataframes
    joint_dataframe = pd.concat(dataframes, ignore_index=True).reset_index(drop=True)

    # Filter out empty masks
    joint_dataframe = joint_dataframe[joint_dataframe.privacy_mask.str.len() > 0].reset_index(drop=True)

    return joint_dataframe


def analyze_dataset(dataframe):
    """Analyze and print dataset statistics."""
    print("\nDataset Analysis:")
    print("-" * 40)
    print(f"Total samples: {len(dataframe)}")
    print("\nSource distribution:")
    print(dataframe["source"].value_counts())

    # Extract all unique labels
    labels = set()
    for row in dataframe["privacy_mask"]:
        for mask in row:
            labels.add(mask["label"])

    print(f"\nUnique labels ({len(labels)}):")
    print(sorted(labels))


def main():
    parser = argparse.ArgumentParser(description="Create evaluation dataset from PII masking datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ai4privacy", "gretel", "both"],
        default=["both"],
        help="Datasets to include (default: both)",
    )
    parser.add_argument("--num-samples", type=int, help="Number of samples per dataset (default: all)")
    parser.add_argument(
        "--output",
        type=str,
        default="_eval_dataset.csv",
        help="Output filename (default: data/eval_dataset.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "pkl"],
        default="csv",
        help="Output format (default: csv)",
    )

    args = parser.parse_args()

    # Handle 'both' option
    if "both" in args.datasets:
        dataset_names = ["ai4privacy", "gretel"]
    else:
        dataset_names = args.datasets

    joint_dataframe = load_and_process_datasets(dataset_names, args.num_samples)

    analyze_dataset(joint_dataframe)

    if args.format == "csv":
        joint_dataframe.to_csv(args.output, index=False)
    else:
        joint_dataframe.to_pickle(args.output)

    print(f"\nDataset saved to: {args.output}")


if __name__ == "__main__":
    main()
