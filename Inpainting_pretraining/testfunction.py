"""
testfunction.py

Implements the testing and evaluation logic for the inpainting model.

Main features:
- Loads the best trained model checkpoint for evaluation.
- Runs inference on the test dataset using a sliding window approach for large images.
- Computes loss (perceptual inpainting loss) and PSNR metrics for each batch.
- Logs evaluation metrics to Weights & Biases (wandb).
- Saves predicted images and input images as NIfTI files for later analysis.
- Handles errors gracefully and skips corrupted files.

Functions:
    test(test_loader, model, directory_name, test_csv):
        Evaluates the model on the test set, logs metrics, and saves results.

    safe_load(batch, split="test"):
        Safely loads data from a batch, handling exceptions and returning tensors or None.
"""


import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import wandb
from monai.inferers import SlidingWindowInferer
from loss import *
from load_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(test_loader, model, directory_name, test_csv):
    """
        Evaluate the trained model on a test dataset and save predictions.

    Args:
        test_loader (ThreadDataLoader): DataLoader for the test set.
        model (torch.nn.Module): The trained Swin UNetR model to be evaluated.
        directory_name (str): Path to the directory where test results will be saved.
        test_csv (str): Path to the CSV file containing image paths for the test set.

    Returns:
        None
    """
    device = next(model.parameters()).device
    # Metrics and loss
    psnr_metric = PSNRMetric(max_val=1.0)
    # Ensure 'test_results' is treated as a string
    os.makedirs(os.path.join(directory_name, "test_results"), exist_ok=True)
    results_dir = os.path.join(directory_name, "test_results")


    # Load the best model
    best_model_path = os.path.join(directory_name, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Best model loaded for testing.")
    else:
        print("Best model not found. Using current model.")

    model.eval()

   # Load the test CSV to get the image paths
    test_df = pd.read_csv(test_csv)

    # Set up sliding window inferer to handle large input images
    window_size = (128, 160, 128)  # Adjust the size as needed to reduce memory consumption

    overlap = .50
    inferer = SlidingWindowInferer(
        roi_size=window_size,
        sw_batch_size=3,
        overlap=overlap,
        mode='gaussian',  # Gaussian blending mode
        device=device
    )

    total_loss = 0.0  # Variable to accumulate the loss
    total_batches = 0  # Counter for the number of batches
    epoch_psnr_scores = []

    with torch.no_grad():  # Ensure inference is done without gradient tracking
        for batch_idx, batch in enumerate(test_loader):
            img, groundtruth, mask = safe_load(batch, split="test")  # Ensure to load nonnoisyage
            if img is None or groundtruth is None or mask is None:
                continue

            img_path = test_df.iloc[batch_idx]['imgs']
            save_filename = os.path.basename(img_path)
            save_path = os.path.join(results_dir, save_filename)  # Use results_dir to save
            save_img_path = os.path.join(results_dir, f"{os.path.splitext(save_filename)[0]}_img.nii.gz")
        
            try:
                img, groundtruth, mask = img.to(device), img.to(device), mask.to(device)
                

                # Perform sliding window inference
                pred_img = inferer(img, model)
                
                #loss = ssim_loss_function(pred_img, groundtruth)  # Use nonnoisyage as ground truth
                loss = perceptual_inpainting_loss_function(pred_img, groundtruth)
                # Update metrics
                psnr_score = psnr_metric(pred_img, groundtruth)
                epoch_psnr_scores.append(psnr_score)
                
                
                
                # Accumulate the loss and count the batch
                total_loss += loss.item()
                total_batches += 1

                # Log evaluation metrics to Weights & Biases
                wandb.log({"Test Loss": loss.item()})
                # wandb.log({
                #    "test_psnr": float(avg_testpsnr) if isinstance(avg_testpsnr, torch.Tensor) else avg_testpsnr,
                # })

                # Save predictions as NIfTI
                pred_img_np = pred_img.squeeze().cpu().numpy()
                
                img_np = img[0].cpu().numpy().squeeze()
                nii_img = nib.Nifti1Image(pred_img_np, np.eye(4))
                nib.save(nib.Nifti1Image(img_np, np.eye(4)), save_img_path)
                nib.save(nii_img, save_path)
                print(f"Test result saved successfully: {save_path}")

            except Exception as e:
                print(f"Error during prediction or saving result: {e}")
        
    # Calculate and log the average loss
    if total_batches > 0:
        average_loss = total_loss / total_batches
        print(f"Average Test Loss: {average_loss}")
        wandb.log({"Average Test Loss": average_loss})
    else:
        print("No valid batches for testing.")
    avg_testpsnr = torch.mean(torch.stack(epoch_psnr_scores)).item() if epoch_psnr_scores else 0
    wandb.log({
   "test_psnr": float(avg_testpsnr) if isinstance(avg_testpsnr, torch.Tensor) else avg_testpsnr,
})

    print(f"PSNR: {avg_testpsnr:.4f}")


def safe_load(batch, split="test"):
    """
    Safely load data from a batch, handling any exceptions that may occur.

    Args:
        batch (dict): A batch of data from the DataLoader, including "img", "mask", and "age" or "nonnoisyage" keys.
        split (str): Specifies the dataset split ("train" or "test") to load the appropriate age label.

    Returns:
        tuple: Tensors for image, age (or nonnoisy age), and mask. Returns (None, None, None) if loading fails.
    """
    try:
        img = batch.get("img", None).to(device)
        groundtruth = batch.get("img", None).to(device)
        mask = batch.get("mask", None).to(device)
        
        return img, groundtruth, mask
    except (EOFError, OSError, ValueError) as e:
        print(f"Skipping corrupted file {batch['img_meta_dict']['filename_or_obj']}: {e}")
        return None, None, None
