"""
 swinunetrcrossentropy.py

Main script for 3D brain image segmentation using the SwinUNETR architecture with cross-entropy loss.

Features:
- Sets up environment, device, and experiment tracking with Weights & Biases (wandb).
- Loads pretrained weights from the inpainting task for transfer learning.
- Initializes SwinUNETR for multi-class segmentation.
- Loads training on 2x total samples, validation (52), and test datasets (52) using MONAI transforms.
- Configures optimizer and learning rate scheduler.
- Supports checkpointing and resuming training.
- Runs training and evaluation loops.
- Optionally supports encoder freezing for decoder-only training.
Usage:
    Run this script to train and evaluate a SwinUNETR model for brain image segmentation.

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
    SEG_PRETRAINED_DATA_CSV, SEG_PRETRAINED_OUTPUT_DIR, TRAIN_SAMPLES_PER_SEX,SEG_PRETRAINED_PROJECT_NAME,
    OPTIMIZER_CLASS, OPTIMIZER_PARAMS, SCHEDULER_CLASS, SCHEDULER_PARAMS, MAX_EPOCHS, TEST_SAMPLES_PER_GROUP, VAL_SAMPLES_PER_GROUP, CACHE_RATE, NUM_WORKERS, BATCH_SIZE, SELECTED_AGE_GROUPS
)

print_config()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if wandb.run is None:
    wandb.init(project=SEG_PRETRAINED_PROJECT_NAME, settings=wandb.Settings(start_method="fork"))
    wandb.run.name = 'x'

def main():
    directory_name = SEG_PRETRAINED_OUTPUT_DIR
    os.makedirs(directory_name, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=(128, 160, 128),
        in_channels=1,
        out_channels=61,
        use_checkpoint=True,
    )
    model = model.to(device)
    optimizer_class = OPTIMIZER_CLASS
    optimizer_params = OPTIMIZER_PARAMS
    scheduler_class = SCHEDULER_CLASS
    scheduler_params = SCHEDULER_PARAMS
    max_epochs = MAX_EPOCHS
    # Load the last model weights (if available) 
    # NOTE: The pretrained weights (e.g., best_model.pth) must exist in `directory_name`
    model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss = load_last_model(model, optimizer_class, optimizer_params, scheduler_class, scheduler_params, directory_name)
    # NOTE: train_samples_total is the number of samples per sex.
    # The total number of training samples will be 2 * train_samples_total.
    train_samples_total = TRAIN_SAMPLES_PER_SEX
    # Load data
    ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data(
        output_dir=SEG_PRETRAINED_OUTPUT_DIR,
        full_data_path=SEG_PRETRAINED_DATA_CSV,
        test_samples_per_group=TEST_SAMPLES_PER_GROUP,
        val_samples_per_group=VAL_SAMPLES_PER_GROUP,
        train_samples_total=TRAIN_SAMPLES_PER_SEX,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
        cache_rate=CACHE_RATE,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        selected_age_groups=SELECTED_AGE_GROUPS,
    )
    train_csv= os.path.join(directory_name, "train_set.csv")
    # Start training
    print("Starting training...")
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

# def main():
#     directory_name = "segmentationTinpaintingencoderfreezing_79S"
#     os.makedirs(directory_name, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = SwinUNETR(
#         img_size=(128, 160, 128),
#         in_channels=1,
#         out_channels=61,
#         use_checkpoint=True,
#     )
#     model = model.to(device)

#     # ✅ Freeze the encoder (SwinViT)
#     # ✅ Freeze both SwinViT and UnetrBasicBlocks (full encoder)
#     for param in model.swinViT.parameters():
#         param.requires_grad = False

#     for param in model.encoder1.parameters():
#         param.requires_grad = False

#     for param in model.encoder2.parameters():
#         param.requires_grad = False

#     for param in model.encoder3.parameters():
#         param.requires_grad = False

#     for param in model.encoder4.parameters():
#         param.requires_grad = False

#     for param in model.encoder10.parameters():
#         param.requires_grad = False  # This is the deepest encoder layer


#     # ✅ Define optimizer and scheduler parameters (for decoder only)
#    optimizer_class = OPTIMIZER_CLASS
#    optimizer_params = OPTIMIZER_PARAMS
#    scheduler_class = SCHEDULER_CLASS
#    scheduler_params = SCHEDULER_PARAMS
#    max_epochs = MAX_EPOCHS

#     # ✅ Create optimizer for decoder only
#     optimizer = optimizer_class(
#         filter(lambda p: p.requires_grad, model.parameters()),  
#         **optimizer_params
#     )

#     # Call load_last_model with correct parameters
#     model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss = load_last_model(
#         model,
#         optimizer_class,
#         optimizer_params,
#         scheduler_class,
#         scheduler_params,  # ✅ Make sure this parameter is passed
#         directory_name     # ✅ Make sure this parameter is passed
#     )

    # train_samples_total = TRAIN_SAMPLES_PER_SEX
    # # Load data
    # ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data(
    #     output_dir=SEG_PRETRAINED_OUTPUT_DIR,
    #     full_data_path=SEG_PRETRAINED_DATA_CSV,
    #     test_samples_per_group=TEST_SAMPLES_PER_GROUP,
    #     val_samples_per_group=VAL_SAMPLES_PER_GROUP,
    #     train_samples_total=TRAIN_SAMPLES_PER_SEX,
    #     train_transforms=train_transforms,
    #     val_transforms=val_transforms,
    #     test_transforms=test_transforms,
    #     cache_rate=CACHE_RATE,
    #     num_workers=NUM_WORKERS,
    #     batch_size=BATCH_SIZE,
    #     selected_age_groups=SELECTED_AGE_GROUPS,
    # )
#     train_csv = os.path.join(directory_name, "train_set.csv")
#     max_epochs = 200  # Set based on your experiment


#     # ✅ Train model
#     train(train_loader, train_csv, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=start_epoch)

#     # ✅ Test model
#     test_csv_path = os.path.join(directory_name, "test_set.csv")
#     test(test_loader, model, directory_name, test_csv_path)

# if __name__ == "__main__":
#     main()
