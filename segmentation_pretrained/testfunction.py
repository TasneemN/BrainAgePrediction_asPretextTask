"""
testfunction.py

Implements the testing and evaluation logic for 3D brain image segmentation using the SwinUNETR architecture.

Main features:
- Loads the best trained model checkpoint for evaluation.
- Runs inference on the test dataset using a sliding window approach for large images.
- Computes cross-entropy loss and Dice score for each batch.
- Logs evaluation metrics (average loss, average Dice, Dice standard deviation) to Weights & Biases (wandb).
- Saves original images, predicted segmentations, and ground truth masks as NIfTI files with correct affine and header information.
- Handles device configuration for GPU/CPU.

Functions:
    test(test_loader, model, directory_name, test_csv):
        Evaluates the model on the test set, logs metrics, and saves results.
"""

import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import wandb
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from torch.nn import CrossEntropyLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose, EnsureType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_loader, model, directory_name, test_csv):
    device = next(model.parameters()).device
    results_dir = os.path.join(directory_name, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    # Load the best model
    best_model_path = os.path.join(directory_name, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Best model loaded for testing.")
    else:
        print("Best model not found. Using current model.")

    model.eval()
    test_df = pd.read_csv(test_csv)
    inferer = SlidingWindowInferer(roi_size=(128, 160, 128), sw_batch_size=2, overlap=0.5, padding_mode="constant")
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=61)])
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=61)])
    criterion = CrossEntropyLoss()
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)
  
    total_loss = 0.0
    total_batches = 0
    dice_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img = batch["img"].to(device)
            seg = batch["seg"].to(device)
            if seg.ndim == 4:
                seg = seg.unsqueeze(1)
            pred_seg = inferer(img, model)
            pred_seg_onehot = [post_pred(i) for i in decollate_batch(pred_seg)]
            seg_onehot = [post_label(i) for i in decollate_batch(seg)]

            # Compute Dice
            dice_metric(y_pred=pred_seg_onehot, y=seg_onehot)
            dice_scores.append(dice_metric.aggregate().item())
            dice_metric.reset()

            # Compute CrossEntropyLoss
            loss = criterion(pred_seg, seg.squeeze(1).long())
            total_loss += loss.item()
            total_batches += 1

 
            img_np = img[0].cpu().numpy().squeeze().astype(np.float32)  # Original Image
            groundtruth_np = seg[0].cpu().numpy().squeeze().astype(np.float32)  # Ground Truth
            pred_seg_np = pred_seg.argmax(dim=1)[0].cpu().numpy().squeeze().astype(np.float32)  # Prediction

            # ✅ Retrieve original image path from `test_df`
            img_path = test_df.iloc[batch_idx]['imgs']

            # ✅ Construct filenames for saving results
            save_filename = os.path.basename(img_path)
            save_img_path = os.path.join(results_dir, f"{os.path.splitext(save_filename)[0]}_original.nii.gz")
            save_pred_path = os.path.join(results_dir, f"{os.path.splitext(save_filename)[0]}_pred.nii.gz")
            save_groundtruth_path = os.path.join(results_dir, f"{os.path.splitext(save_filename)[0]}_groundtruth.nii.gz")

            # ✅ Load correct affine matrix from the original image
           
            original_nifti = nib.load(img_path)  # Load the original image
            affine = original_nifti.affine  # Get the correct affine matrix
            header = original_nifti.header  # ✅ Copy the header information
            target_shape = original_nifti.shape  # Get the correct shape

            # ✅ Save Original Image, Prediction, and Ground Truth as NIfTI files with the header
            nib.save(nib.Nifti1Image(img_np, affine, header=header), save_img_path)
            nib.save(nib.Nifti1Image(pred_seg_np, affine, header=header), save_pred_path)
            nib.save(nib.Nifti1Image(groundtruth_np, affine, header=header), save_groundtruth_path)

            print(f"✅ Test results saved: {save_img_path}, {save_pred_path}, {save_groundtruth_path}")

    # Calculate average Dice score and standard deviation
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    std_dice = np.std(dice_scores, ddof=1) if len(dice_scores) > 1 else 0.0  # ddof=1 for sample standard deviation

    # Calculate average loss
    avg_loss = total_loss / total_batches if total_batches > 0 else 0

    # Print results
    print(f"Average Test Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}, Std Dice: {std_dice:.4f}")

    # Log results to WandB
    wandb.log({
        "Average Test Loss": avg_loss,
        "Average Dice": avg_dice,
        "Dice Standard Deviation": std_dice
    })

