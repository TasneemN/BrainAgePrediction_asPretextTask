"""
train.py

Training script for 3D brain image segmentation from scratch using the SwinUNETR architecture and cross-entropy loss.

Main features:
- Trains the model using MONAI transforms and metrics.
- Computes cross-entropy loss and Dice score for multi-class segmentation.
- Logs training and validation losses, Dice scores, and learning rate to Weights & Biases (wandb).
- Saves best and last model checkpoints during training.
- Optionally saves training patches as NIfTI files for debugging.
- Handles device configuration for GPU/CPU.


Functions:
    train(train_loader, train_csv, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
        Main training loop that iterates over epochs and batches, logs metrics, and saves checkpoints.
"""


import torch
import os
import pandas as pd
import numpy as np
import wandb
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from monai.data import decollate_batch
import nibabel as nib
from monai.transforms import AsDiscrete, Compose, EnsureType

from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_loader, train_csv, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
    """
    Training function for SwinUNETR model with MONAI transforms and metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
        # Load the training CSV to get the image paths
    train_df = pd.read_csv(train_csv)
    
    # Loss and metrics
    criterion = CrossEntropyLoss()
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)

    # Post-processing transforms
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=61)])  # Adjusted for 61 classes
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=61)])

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(directory_name, "best_model.pth")
    last_model_path = os.path.join(directory_name, "last_model.pth")
    training_results_dir = os.path.join(directory_name, "training")
    os.makedirs(directory_name, exist_ok=True)
    os.makedirs(training_results_dir, exist_ok=True)  # Ensure the training results directory exists

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Loaded checkpoint with best validation loss: {best_val_loss}")

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            img, seg = batch["img"].to(device), batch["seg"].to(device)

            # Ensure seg has a channel dimension
            if seg.ndim == 4:
                seg = seg.unsqueeze(1)

            optimizer.zero_grad()
            pred_seg = model(img)
            loss = criterion(pred_seg, seg.squeeze(1).long())
            loss.backward()
            optimizer.step()
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            train_loss += loss.item()
                        # Save patches every 10 batches for debugging
            if batch_idx % 10 == 0:
                try:
                    # Extract paths from the DataFrame
                    img_path = train_df.iloc[batch_idx]['imgs']
                    seg_path = train_df.iloc[batch_idx]['seg']

                    # Extract filenames
                    save_filename = os.path.basename(img_path)
                    save_segname = os.path.basename(seg_path)

                    # Convert to numpy for saving
                    img_np = img[0].cpu().numpy().squeeze().astype(np.float32)
                    seg_np = seg[0].cpu().numpy().squeeze().astype(np.float32)
                    pred_seg_np = pred_seg.argmax(dim=1)[0].cpu().numpy().squeeze().astype(np.float32)

                    # Save paths
                    save_img_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_batch{batch_idx}_img.nii.gz")
                    save_seg_path = os.path.join(training_results_dir, f"{os.path.splitext(save_segname)[0]}_batch{batch_idx}_seg.nii.gz")
                    save_pred_path = os.path.join(training_results_dir, f"{os.path.splitext(save_filename)[0]}_batch{batch_idx}_pred.nii.gz")

                    # Save NIfTI files
                    affine = nib.load(img_path).affine
                    nib.save(nib.Nifti1Image(img_np, affine), save_img_path)
                    nib.save(nib.Nifti1Image(seg_np, affine), save_seg_path)
                    nib.save(nib.Nifti1Image(pred_seg_np, affine), save_pred_path)

                    print(f"Saved training patch (epoch {epoch}, batch {batch_idx}): {save_img_path}, {save_seg_path}, {save_pred_path}")
                except Exception as e:
                    print(f"Error saving patch for batch {batch_idx}: {e}")

            

        train_loss /= len(train_loader)
        print(f"Epoch {epoch} Training Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        dice_scores = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                img, seg = batch["img"].to(device), batch["seg"].to(device)

                # Ensure seg has a channel dimension
                if seg.ndim == 4:
                    seg = seg.unsqueeze(1)

                pred_seg = model(img)
                loss = criterion(pred_seg, seg.squeeze(1).long())
                val_loss += loss.item()

                pred_seg_onehot = [post_pred(i) for i in decollate_batch(pred_seg)]
                seg_onehot = [post_label(i) for i in decollate_batch(seg)]
                dice_metric(y_pred=pred_seg_onehot, y=seg_onehot)
                dice_scores.append(dice_metric.aggregate().item())
                dice_metric.reset()

        val_loss /= len(val_loader)
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else float("nan")
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}, Avg Dice: {avg_dice:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "dice_scores": dice_scores, "avg_dice": avg_dice})

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"Best model saved at epoch {epoch}")

        # Save last model
        torch.save({
            'epoch': epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_loss
        }, last_model_path)

        scheduler.step()
        

    print("Training complete.")


# def safe_load(batch):
#     """
#     Safely load data from a batch, handling any exceptions that may occur.
#     """
#     try:
#         img = batch.get("img", None)
#         seg = batch.get("seg", None)

#         img = img.to(device) if img is not None and isinstance(img, torch.Tensor) else None
#         seg = seg.to(device) if seg is not None and isinstance(seg, torch.Tensor) else None

#         return img, seg
#     except (EOFError, OSError, ValueError, AttributeError) as e:
#         print(f"Skipping corrupted file: {e}")
#         return None, None
