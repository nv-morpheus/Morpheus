#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import json
import numpy as np
import pandas as pd
import pytest
import torch

from morpheus.models.dfencoder.autoencoder import AutoEncoder
from morpheus.models.dfencoder.dataloader import DatasetFromPath, get_distributed_training_dataloader_from_path, get_validation_dataset_from_path
from utils import TEST_DIRS

INFERENCE_START_DATE = "2022-11-01"
VALIDATION_SET_SIZE = 3000  # around 1/10 of the train set
TRAIN_VAL_SPLIT_SEED = 11

FEATURE_COLUMNS = [
    "app_name",
    "browser_type",
    "os",
    "has_error",
    "country",
    "city",
    "log_count",
    "location_incr",
    "app_incr",
]
LOSS_TYPES = ["train", "val", "id_val"]
# 75th quantile of the losses from 100 times of offline training
LOSS_TARGETS = {
    "train": {
        "log_count": 0.33991,
        "location_incr": 0.30789,
        "app_incr": 0.17698,
        "has_error": 0.00878,
        "app_name": 0.13066,
        "browser_type": 0.39804,
        "os": 0.09882,
        "country": 0.06063,
        "city": 0.32344,
    },
    "val": {
        "log_count": 0.3384,
        "location_incr": 0.31456,
        "app_incr": 0.16201,
        "has_error": 0.00614,
        "app_name": 0.11907,
        "browser_type": 0.38239,
        "os": 0.00064,
        "country": 0.0042,
        "city": 0.32161,
    },
    "id_val": {
        "log_count": 0.07079,
        "location_incr": 0.05318,
        "app_incr": 0.03659,
        "has_error": 0.0046,
        "app_name": 0.03542,
        "browser_type": 0.0915,
        "os": 0.00057,
        "country": 0.00343,
        "city": 0.08525,
    },
}
LOSS_TOLERANCE_RATIO = 1.1

TEST_DATA_DIR = os.path.join(TEST_DIRS.validation_data_dir, "dfp_dataloader_test")
TRAIN_FOLDER = os.path.join(TEST_DATA_DIR, "train")
VALIDATION_FOLDER = os.path.join(TEST_DATA_DIR, "validation")
INFERENCE_FOLDER = os.path.join(TEST_DATA_DIR, "inference")
PRESET_CATS_FILEPATH = os.path.join(TEST_DATA_DIR, 'preset_cats.json')
PRESET_NUMERICAL_SCALER_PARAMS_FILEPATH = os.path.join(TEST_DATA_DIR, 'preset_numerical_scaler_params.json')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def test_dfencoder_distributed_e2e():
    world_size = 1
    torch.multiprocessing.spawn(_run_test, args=(world_size,), nprocs=world_size, join=True)    

@pytest.mark.usefixtures("manual_seed")
def _run_test(rank, world_size):
    torch.cuda.set_device(rank)

    setup(rank, world_size)

    preset_cats = json.load(open(PRESET_CATS_FILEPATH, 'r'))
    preset_numerical_scaler_params = json.load(open(PRESET_NUMERICAL_SCALER_PARAMS_FILEPATH, 'r'))
    
    # Initializing model
    model = AutoEncoder(
        encoder_layers = [512, 500],
        decoder_layers = [512],
        activation='relu',
        swap_p=0.2,
        lr = 0.01, 
        lr_decay=0.99,
        batch_size=4096,
        logger='basic', 
        verbose=True,
        progress_bar=False,
        optimizer='adam', 
        scaler='standard',
        min_cats=1, 
        device=rank,
        preset_numerical_scaler_params=preset_numerical_scaler_params,
        binary_feature_list=[],
        preset_cats=preset_cats,
        eval_batch_size=1024,
        patience=5,
        distributed_training=True,
    )

    # Prepare the dataloader
    dataloader = get_distributed_training_dataloader_from_path(
        model, data_folder=TRAIN_FOLDER, rank=rank, world_size=world_size)
    # Load validation set
    val_dataset = get_validation_dataset_from_path(model, VALIDATION_FOLDER)

    # Train
    model.fit(
        train_data=dataloader, 
        rank=rank, 
        world_size=world_size, 
        epochs=10,
        val_data=val_dataset, 
        run_validation=True, 
        use_val_for_loss_stats=True
    )

    if rank == 0: 
        # Make sure model converges (low loss)
        for loss_type in LOSS_TYPES:
            ft_losses = getattr(model.logger, f"{loss_type}_fts")
            for ft, losses_l in ft_losses.items():
                losses = losses_l[1]
                assert min(losses) < LOSS_TARGETS[loss_type][ft] * LOSS_TOLERANCE_RATIO

        # Inference
        inf_dataset = DatasetFromPath(
            data_folder=INFERENCE_FOLDER,
            batch_size=1024,
            preprocess_fn=model.preprocess_validation_data, 
            shuffle_rows_in_batch=False,
            preload_data_into_memory=True, # very small inference set
        )
        inf_pdf = inf_dataset.get_preloaded_data()
        inf_res = model.get_results_from_dataset(inf_dataset, preloaded_df=inf_pdf, return_abs=True)

        # Assert the consistency of output rows and columns
        assert len(inf_res) == len(inf_pdf)
        assert sorted(inf_res.columns) == sorted(
            [ft + col_suffix for ft in FEATURE_COLUMNS
            for col_suffix in ["", "_pred", "_loss", "_z_loss"]] + ["max_abs_z", "mean_abs_z", "z_loss_scaler_type"])
        # make sure the user baseline is modeled well enough so the minimum and median z scores
        # from inference are in range
        assert min(inf_res.mean_abs_z) < 1
        assert (np.median(inf_res.mean_abs_z) < 100
                )  # expect median mean_abs_z to be < 50. Using 100 to leave some room for variability

    cleanup()