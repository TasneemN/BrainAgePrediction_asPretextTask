"""
testfunction.py

Implements the testing and evaluation logic for voxel-level brain age prediction.

Main features:
- Loads the best trained model checkpoint for evaluation.
- Runs inference on the test dataset using a sliding window approach for large images.
- Computes loss (voxel-level MAE) for each batch, focusing on masked (brain) regions.
- Logs evaluation metrics (mean and standard deviation of test loss) to Weights & Biases (wandb).
- Saves predicted age maps as NIfTI files for later analysis.
- Handles errors gracefully and skips corrupted files.

Functions:
    test(test_loader, model, directory_name, test_csv):
        Evaluates the model on the test set, logs metrics, and saves results.

    safe_load(batch, split="train"):
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
    test_losses = []  # ✅ Store losses for standard deviation calculation

    
    with torch.no_grad():  # Ensure inference is done without gradient tracking
        for batch_idx, batch in enumerate(test_loader):
            img, nonnoisyage, mask = safe_load(batch, split="test")  # Ensure to load nonnoisyage
            if img is None or nonnoisyage is None or mask is None:
                continue

            img_path = test_df.iloc[batch_idx]['imgs']
            save_filename = os.path.basename(img_path)
            save_path = os.path.join(results_dir, save_filename)  # Use results_dir to save

            try:
                img, nonnoisyage, mask = img.to(device), nonnoisyage.to(device), mask.to(device)
                masked_img = img * mask
                
                # Perform sliding window inference
                pred_age = inferer(masked_img, model)
                pred_age = pred_age * mask
                loss = voxel_mae(pred_age, nonnoisyage, mask)  # Use nonnoisyage as ground truth
                
                # Accumulate the loss and count the batch
                total_loss += loss.item()
                ############################
                test_losses.append(loss.item())  # ✅ Store loss for standard deviation calculation

                total_batches += 1
                

                # Log evaluation metrics to Weights & Biases
                #wandb.log({"Test Loss": loss.item()})
                # ✅ Log test loss with manual test step counter (starting from 0)
                wandb.log({"Test Loss": loss.item()})
                print(f"Test Loss: {loss.item()}")
                # Save predictions as NIfTI
                pred_age_np = pred_age.squeeze().cpu().numpy()
                nii_img = nib.Nifti1Image(pred_age_np, np.eye(4))
                nib.save(nii_img, save_path)
                print(f"Test result saved successfully: {save_path}")

            except Exception as e:
                print(f"Error during prediction or saving result: {e}")
      # ✅ Compute Mean and Standard Deviation of Test Loss
    if total_batches > 0:
        average_loss = total_loss / total_batches
        std_test_loss = np.std(test_losses)  # ✅ Compute standard deviation
        print(f"Average Test Loss: {average_loss:.4f} ± {std_test_loss:.4f}")
        
        # ✅ Log metrics to Weights & Biases
        wandb.log({
            "Average Test Loss": average_loss,
            "Test Loss Std Dev": std_test_loss  # ✅ Log standard deviation
        })    
    # # Calculate and log the average loss
    # if total_batches > 0:
    #     average_loss = total_loss / total_batches
    #     print(f"Average Test Loss: {average_loss}")
    #     wandb.log({"Average Test Loss": average_loss})
    else:
        print("No valid batches for testing.")


def safe_load(batch, split="train"):
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
        mask = batch.get("mask", None).to(device)
        if split == "train":
            age = batch.get("age", None).to(device)
        else:
            age = batch.get("nonnoisyage", None).to(device)
        return img, age, mask
    except (EOFError, OSError, ValueError) as e:
        print(f"Skipping corrupted file {batch['img_meta_dict']['filename_or_obj']}: {e}")
        return None, None, None
