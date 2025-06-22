"""
train.py

Contains the training loop for the SwinUNETR model for inpainting pretraining.

This script handles:
- Model training and validation.
- Logging metrics using Weights & Biases (wandb).
- Saving model checkpoints.
- Loading data using MONAI transforms.
- Computing perceptual loss for inpainting tasks.
- Saving predictions as NIfTI files.
- Handling device configuration for GPU/CPU.

Main functions:
    train(train_loader, train_csv, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
        Main training loop that iterates over epochs and batches.

    load_last_model(model, optimizer, scheduler, directory_name, reset_lr=None):
        Loads the last saved model checkpoint to resume training if available.
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
from monai.metrics import PSNRMetric
from load_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_loader, train_csv, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
    """
    Training function that uses a train CSV file to load and log additional information.
    """
    # Ensure the model and tensors are moved to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Load the training CSV to get additional information
    train_df = pd.read_csv(train_csv)

    # Metrics and loss
    psnr_metric = PSNRMetric(max_val=1.0)

    # Directories for saving results
    checkpoint_path = os.path.join(directory_name, "best_model.pt")
    last_model_path = os.path.join(directory_name, "last_model.pt")
    training_results_dir = os.path.join(directory_name, "training")
    os.makedirs(directory_name, exist_ok=True)
    os.makedirs(training_results_dir, exist_ok=True)

    # Load best validation loss if a checkpoint exists
    best_val_loss = float("inf")
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Last model loaded. Resuming training from epoch: {best_val_loss}")

    for epoch in range(start_epoch, max_epochs + 1):
        print(f"Epoch {epoch}")
        model.train()  # Ensure model is in training mode
        train_loss = 0.0
        epoch_psnr_scores = []

        for batch_idx, batch in enumerate(train_loader):
            img, groundtruth = batch["img"].to(device), batch["groundtruth"].to(device)
            optimizer.zero_grad()
                            
               # Forward pass
            pred_img = model(img).float()

            # Ensure predicted image requires gradients
            assert pred_img.requires_grad, "Predicted image does not require gradients!"

            # Compute loss with gradients
            #loss = combined_inpainting_loss(pred_img, groundtruth, is_training= True)  # Pass train=True to enable gradients
            loss = perceptual_inpainting_loss_function(pred_img, groundtruth, is_training= True)  # Compute loss  # Gradients enabled by default           

          
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            print("=", end = "")

            train_loss += loss.item()

            # Update metrics
            psnr_score = psnr_metric(pred_img, groundtruth)
            epoch_psnr_scores.append(psnr_score)

            # Save intermediate results every 10 batches (for debugging only)
            if batch_idx % 200 == 0:
                try:
                    img_path = train_df.iloc[batch_idx]['imgs']
                    save_filename = os.path.basename(img_path)
                    img_np = img[0].cpu().numpy().squeeze()
                    groundtruth_np = groundtruth[0].cpu().numpy().squeeze()
                    pred_img_np = pred_img[0].detach().cpu().numpy().squeeze()

                    # Save paths
                    save_img_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_epoch{epoch}_img.nii.gz")
                    save_pred_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_epoch{epoch}_pred.nii.gz")
                    save_groundtruth_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_epoch{epoch}_groundtruth.nii.gz")

                    affine = nib.load(img_path).affine
                    nib.save(nib.Nifti1Image(img_np, affine), save_img_path)
                    nib.save(nib.Nifti1Image(pred_img_np, affine), save_pred_path)
                    nib.save(nib.Nifti1Image(groundtruth_np, affine), save_groundtruth_path)

                    print(f"Saved training patch: {save_img_path}, {save_groundtruth_path}, {save_pred_path}")
                except Exception as e:
                    print(f"Error saving patch for batch {batch_idx}: {e}")

        train_loss /= len(train_loader)
        avg_psnr = torch.mean(torch.stack(epoch_psnr_scores)).item() if epoch_psnr_scores else 0

        print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}, PSNR: {avg_psnr:.4f}")

    
          # Validation phase
        print(f"Starting Validation for Epoch {epoch}")
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        epoch_val_psnr_scores = []

        with torch.no_grad():  # No gradients needed for validation
            for batch_idx, batch in enumerate(val_loader):
                
                img, groundtruth= batch["img"].to(device), batch["groundtruth"].to(device)

                # Forward pass
                pred_img = model(img).float()

                
                # Compute Validation Loss
                val_loss = perceptual_inpainting_loss_function(pred_img, groundtruth, is_training= False)

                val_loss += loss.item()

                # Compute PSNR
                psnr_val_score = psnr_metric(pred_img, groundtruth)
                epoch_val_psnr_scores.append(psnr_val_score)
                print(f"Validation Batch {batch_idx}: pred_img shape {pred_img.shape}, groundtruth shape {groundtruth.shape}")


                # Save intermediate results every 10 batches (for debugging only)   
                if batch_idx % 100 == 0:
                    try:
                        img_path = train_df.iloc[batch_idx]['imgs']
                        save_filename = os.path.basename(img_path)
                        img_np = img[0].cpu().numpy().squeeze()
                        groundtruth_np = groundtruth[0].cpu().numpy().squeeze()
                        pred_img_np = pred_img[0].detach().cpu().numpy().squeeze()

                        # Save paths
                        save_img_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_val_epoch{epoch}_img.nii.gz")
                        save_pred_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_val_epoch{epoch}_pred.nii.gz")
                        save_groundtruth_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_val_epoch{epoch}_groundtruth.nii.gz")

                        affine = nib.load(img_path).affine
                        nib.save(nib.Nifti1Image(img_np, affine), save_img_path)
                        nib.save(nib.Nifti1Image(pred_img_np, affine), save_pred_path)
                        nib.save(nib.Nifti1Image(groundtruth_np, affine), save_groundtruth_path)

                        print(f"Saved validation patch: {save_img_path}, {save_groundtruth_path}, {save_pred_path}")
                    except Exception as e:
                        print(f"Error saving patch for batch {batch_idx}: {e}") 
        # Compute average validation loss
        val_loss /= len(val_loader)
        

        # Compute average PSNR for validation
        avg_val_psnr = torch.mean(torch.stack(epoch_val_psnr_scores)).item() if epoch_val_psnr_scores else 0



        print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, PSNR: {avg_val_psnr:.4f}")
        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, " | ", optimizer.param_groups[0]['lr'])
        wandb.log({
            "train_loss": float(train_loss),  # Convert MetaTensor to float
            "val_loss": float(val_loss) if isinstance(val_loss, torch.Tensor) else val_loss,  # Convert only if tensor
            "val_psnr": float(avg_val_psnr) if isinstance(avg_val_psnr, torch.Tensor) else avg_val_psnr,
            "train_psnr": float(avg_psnr) if isinstance(avg_psnr, torch.Tensor) else avg_psnr,
            "lr": optimizer.param_groups[0]['lr']
        })

        # Save the best model

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss
            }, checkpoint_path)
            print("Best model saved.")

        # Save the last model
        torch.save({
            'epoch': epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss
        }, last_model_path)

        # Step the scheduler
        scheduler.step()
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
