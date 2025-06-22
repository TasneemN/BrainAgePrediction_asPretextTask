"""
swinunetr.py

Entry point for the inpainting pretraining project using the SwinUNETR architecture.

This script:
- Sets up the environment and device for training.
- Initializes experiment tracking with Weights & Biases (wandb).
- Defines the main workflow for model training and testing.
- Loads data, model, optimizer, and scheduler.
- Handles checkpoint loading and saving.
- Runs the training and testing phases using external modules.

Modules imported:
- MONAI (medical imaging deep learning framework)
- PyTorch (deep learning framework)
- wandb (experiment tracking)
- Custom modules: transforms, load_data, loss, train, testfunction

Usage:
    Run this script directly to start the full training and testing workflow.

Functions:
    main(): Initializes all components and orchestrates the training and testing process.
"""
import monai
import torch
import wandb
import os
import pandas as pd
import numpy as np
import nibabel as nib
import re


from monai.config import print_config
from monai.networks.nets import SwinUNETR
from transforms import *
from load_data import *
from loss import *
from train import *
from testfunction import *
from config import *
print_config()

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB
if wandb.run is None:
    wandb.init(
        project=SWINUNETR_INPAINTING_PROJECT_NAME,
        settings=wandb.Settings(start_method="fork", _service_wait=300)
    )
    wandb.run.name = 'x'

def main():
    """
    Main function to initialize the model, load data, and run training and testing phases.
    """
    directory_name = SWINUNETR_INPAINTING_DIR
    output_dir = SWINUNETR_INPAINTING_OUTPUT_DIR
    os.makedirs(directory_name, exist_ok=True)

    # Initialize model
    model = SwinUNETR(
        img_size=(128, 160, 128),
        in_channels=1,
        out_channels=1,
        use_checkpoint=True,
    )
    model = torch.nn.DataParallel(model).to(device)

   
    # Initialize optimizer and scheduler from config
    optimizer = SWINUNETR_INPAINTING_OPTIMIZER_CLASS(model.parameters(), **SWINUNETR_INPAINTING_OPTIMIZER_PARAMS)
    scheduler = SWINUNETR_INPAINTING_SCHEDULER_CLASS(optimizer, **SWINUNETR_INPAINTING_SCHEDULER_PARAMS)
    max_epochs = SWINUNETR_INPAINTING_MAX_EPOCHS

    # Load the last model checkpoint
    model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss = load_last_model(
        model, optimizer, scheduler, directory_name
    )

    # Load datasets and data loaders
    ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data(
        directory_name=directory_name,
        output_dir=output_dir
    )

    # Training phase
    train_csv = os.path.join(directory_name, "train_set.csv")
    train(
        train_loader=train_loader,
        train_csv=train_csv,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        directory_name=directory_name,
        start_epoch=start_epoch
    )

    # Testing phase
    test(
        test_loader=test_loader,
        model=model,
        directory_name=directory_name,
        test_csv=TEST_CSV
    )
    print("Training and testing complete.")

if __name__ == "__main__":
    print(f"Using device: {device}")
    main()
