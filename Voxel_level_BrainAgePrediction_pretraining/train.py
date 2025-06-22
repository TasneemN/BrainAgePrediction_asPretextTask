"""
train.py

Contains the training loop and checkpoint management for voxel-level brain age prediction pretraining.

Main features:
- Trains a model using masked input images and voxel-level age maps.
- Computes loss using a custom voxel-level MAE loss function, focusing on masked (brain) regions.
- Logs training and validation losses, as well as learning rate, to Weights & Biases (wandb).
- Saves model checkpoints for the best validation loss and the last epoch.
- Supports resuming training from the last saved checkpoint, with optional learning rate reset.
- Handles device configuration for GPU/CPU.

Functions:
    train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
        Main training loop that iterates over epochs and batches, logs metrics, and saves checkpoints.

    load_last_model(model, optimizer, scheduler, directory_name, reset_lr=None):
        Loads the last saved model checkpoint to resume training if available, with optional learning rate reset.
        """
import monai
import torch
import wandb
import os
import pandas as pd
import numpy as np
import nibabel as nib
import re
from loss import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training function   
def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
    """
    Train a model for a given number of epochs, saving checkpoints for the best model and the last epoch.

    Args:
        train_loader (ThreadDataLoader): DataLoader for the training set.
        val_loader (ThreadDataLoader): DataLoader for the validation set.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        max_epochs (int): Maximum number of training epochs.
        directory_name (str): Directory to save model checkpoints.
        start_epoch (int): The epoch to start training from, useful for resuming training.

    Returns:
        None
    """
    model.train()

    #best_val_loss = float('inf')
        # Define the path to the checkpoint file
    checkpoint_path = os.path.join(directory_name, "best_model.pt")

    # Initialize best_val_loss
    if os.path.exists(checkpoint_path):
        # Try to load the saved checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Default to infinity if key is missing
            print(f"Loaded best_val_loss from checkpoint: {best_val_loss}")
        except Exception as e:
            # If there is an error loading the checkpoint, initialize to infinity
            print(f"Error loading checkpoint: {e}")
            best_val_loss = float('inf')
    else:
        # If checkpoint doesn't exist, initialize to infinity
        print("No checkpoint found. Initializing best_val_loss to infinity.")
        best_val_loss = float('inf')

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        print("Epoch ", epoch)
        print("Train:", end ="")
        if epoch < 50:
            voxel_coef = 1
        elif 50 <= epoch < 130:
            voxel_coef = 1
        else:
            voxel_coef = 1.3
        
        step = 0
        for batch_idx, batch in enumerate(train_loader):
            img, age, mask = batch["img"].to(device), batch["age"].to(device), batch["mask"].to(device)
            file_name = batch.get("file_name", "Unknown file")  # Assuming file names are included in the batch
            
            optimizer.zero_grad()
            
            # Apply the mask to the input image and the ground truth age map
            masked_img = img * mask  # Masked input image
            masked_age = age * mask  # Masked ground truth (age)
            
            # Forward pass
            pred_age = model(masked_img)
            
            # Compute loss
            voxel_mae_value = voxel_mae(pred_age, masked_age, mask)

            # Check if voxel_mae is None (i.e., empty tensor list)
            if voxel_mae_value is None:
                #print(f"The voxel_mae list is empty for the file: {file_name}")
                #print(f"pred_age tensor: {pred_age}")
                #print(f"age tensor: {age}")
                continue  # Skip this batch to avoid errors

            loss = voxel_coef * voxel_mae_value

            # Check if loss requires gradient
            # print(f"loss.requires_grad: {loss.requires_grad}")        

            #loss = voxel_coef * voxel_mae(pred_age, age)
            if loss.requires_grad:
                loss.backward()
            # else:
            #     print("Loss tensor does not require gradients.")

            #loss.backward()
            train_loss += loss.item()
            optimizer.step()
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            print("=", end = "")
            step += 1

        train_loss = train_loss / (step + 1)

        print()
        print("Val:", end = "")
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                #img, age, mask = batch["img"].to(device), batch["age"].to(device), batch["mask"].to(device)
                img, nonnoisyage, mask = batch["img"].to(device), batch["nonnoisyage"].to(device), batch["mask"].to(device)

                
                # Apply the mask to the input image and the ground truth age map
                masked_img = img * mask  # Masked input image
                masked_age = nonnoisyage * mask  # Masked ground truth (nonnoisyage)
  
                # Forward pass for validation
                pred_age = model(masked_img)

                # Compute validation loss
                loss = voxel_coef * voxel_mae(pred_age, masked_age, mask)
                val_loss += loss.item()
                print("=", end="")
        val_loss = val_loss / len(val_loader)

        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, " | ", optimizer.param_groups[0]['lr'])

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # if epoch == 1:
        #     best_val_loss = val_loss

        if val_loss < best_val_loss:
            print("Saving best model")
            best_val_loss = val_loss
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            save_path = os.path.join(directory_name, "best_model.pt")
            torch.save(state, save_path)

        # Save last model weights
        print("Saving last model")
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': val_loss,
        }
        save_path = os.path.join(directory_name, "last_model.pt")
        torch.save(state, save_path)

        # Step the scheduler and log the updated learning rate
        print(f"Before step: {optimizer.param_groups[0]['lr']}")
        scheduler.step()  # Update the learning rate
        torch.cuda.empty_cache()
        print(f"After step: {optimizer.param_groups[0]['lr']}")
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"lr": current_lr})
        print(f"Learning Rate after step: {current_lr}")

    print("Training complete.")

def load_last_model(model, optimizer, scheduler, directory_name, reset_lr=None):
    """
    Load the last saved model checkpoint to resume training if available.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to load state.
        directory_name (str): Directory where model checkpoints are stored.
        reset_lr (float, optional): Optionally reset learning rate to a specified value.

    Returns:
        tuple: Updated model, optimizer, scheduler, start_epoch, last_val_loss, and best_val_loss.
    """
    # Load the last model weights
    last_model_path = os.path.join(directory_name, "last_model.pt")
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        last_val_loss = checkpoint['val_loss']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Get the best val loss from checkpoint if available
        print(f"Last model loaded. Resuming training from epoch {start_epoch}")
        
        # Optionally reset the learning rate if a new learning rate is provided
        if reset_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = reset_lr
            print(f"Learning Rate after resetting: {optimizer.param_groups[0]['lr']}")
        
        return model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss
    else:
        print("Last model weights not found. Starting training from scratch.")
        return model, optimizer, scheduler, 1, float('inf'), float('inf')
