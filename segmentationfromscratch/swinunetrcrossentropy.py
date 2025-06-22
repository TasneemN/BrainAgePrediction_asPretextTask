"""
swinunetrcrossentropy.py

Main script for 3D brain image segmentation from scratch using the SwinUNETR architecture and cross-entropy loss.

Main features:
- Sets up the environment, device, and Weights & Biases (wandb) experiment tracking.
- Initializes the SwinUNETR model for multi-class segmentation.
- Loads training, validation, and test datasets and dataloaders using MONAI transforms.
- Configures optimizer and learning rate scheduler.
- Supports checkpointing and resuming training from the last saved checkpoint.
- Runs the full training loop and evaluates the model on the test set.
- Prints a sample batch from the test loader for verification.

Functions:
    main():
        Initializes all components, runs training, and performs final testing.
"""

import os
import csv
import random
import re
from collections import defaultdict
import torch
import wandb
import pandas as pd
import numpy as np
import nibabel as nib

from torch.nn import CrossEntropyLoss
from transforms import train_transforms, val_transforms, test_transforms
from testfunction import test
from load_data import *
from train import *

from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.config import print_config
from monai.networks.nets import SwinUNETR
from config import (
    SEG_SCRATCH_OUTPUT_DIR, SEG_SCRATCH_PROJECT_NAME, SEG_SCRATCH_DATA_CSV,
    TRAIN_SAMPLES_PER_SEX, OPTIMIZER_CLASS, OPTIMIZER_PARAMS, SCHEDULER_CLASS, SCHEDULER_PARAMS, MAX_EPOCHS
)

print_config()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if wandb.run is None:
    wandb.init(project= SEG_SCRATCH_PROJECT_NAME, settings=wandb.Settings(start_method="fork"))
    wandb.run.name = 'x'

def main():
    directory_name = SEG_SCRATCH_OUTPUT_DIR
    os.makedirs(directory_name, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=(128, 160, 128),
        in_channels=1,
        out_channels=61,
        use_checkpoint=True,
    )
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Initialize optimizer and scheduler
    optimizer = OPTIMIZER_CLASS(model.parameters(), **OPTIMIZER_PARAMS)
    scheduler = SCHEDULER_CLASS(optimizer, **SCHEDULER_PARAMS)
    max_epochs = MAX_EPOCHS

    # Load the last model weights (if available)   
     
    model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss = load_last_model(model, optimizer, scheduler,  directory_name)

    ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data(
            output_dir=directory_name,
            train_samples_total=TRAIN_SAMPLES_PER_SEX,
            full_data_path=SEG_SCRATCH_DATA_CSV,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )
    
        # Call test function with test_csv file path
    train_csv= os.path.join(directory_name, "train_set.csv")
    # Start training
    
    train(train_loader, train_csv, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=start_epoch)

    # Print a sample batch to verify test_loader content
    print("Checking a sample batch from test_loader")
    sample_batch = next(iter(test_loader), None)
    if sample_batch:
        print(f"Sample batch keys: {sample_batch.keys()}")
    else:
        print("Test loader is empty or not loading data correctly.")
    test_csv_path= os.path.join(directory_name, "test_set.csv")
    
    test(test_loader, model, directory_name, test_csv_path)

if __name__ == "__main__":
    main()
