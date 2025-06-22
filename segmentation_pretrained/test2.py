"""
test2.py
Mainly for testing SwinUNETR model on the external test dataset (OASIS2).

Script for evaluating a trained SwinUNETR model on 3D brain image segmentation tasks.

Main features:
- Loads the best trained SwinUNETR model checkpoint for evaluation.
- Loads and preprocesses test data using MONAI transforms.
- Runs inference on the test dataset using a sliding window approach for large images.
- Computes cross-entropy loss and Dice score for each batch.
- Logs evaluation metrics (average loss, average Dice, Dice standard deviation) to Weights & Biases (wandb).
- Saves predicted segmentations as NIfTI files, preserving affine and header information.
- Handles missing files and device configuration for GPU/CPU.

Functions:
    load_test_data(test_csv, test_transforms, batch_size=1, num_workers=4):
        Loads and preprocesses test data from a CSV file and returns a DataLoader.

    test(test_loader, model, directory_name, test_csv_path):
        Performs inference on the test set, logs metrics, and saves results.

    main():
        Sets up the environment, loads the model and data, and runs the test pipeline.
"""

import os
import pandas as pd
import numpy as np
import torch
import nibabel as nib
import wandb
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, ToTensord,
    AsDiscrete, EnsureType
)
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch
from torch.nn import CrossEntropyLoss
from config import SEG_TEST_OUTPUT_DIR, SEG_TEST_PROJECT_NAME, OASIS2_TEST_CSV
# ✅ Test Transforms
test_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
    ToTensord(keys=["img", "seg"]),
])

# ✅ Load Test Data Function
def load_test_data(test_csv, test_transforms, batch_size=1, num_workers=4):
    """
    Loads test data from test_set.csv and prepares DataLoader.

    Args:
        test_csv (str): Path to the test dataset CSV file.
        test_transforms (Compose): Preprocessing transforms for test data.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.

    Returns:
        DataLoader: Test data loader ready for inference.
    """
    test_df = pd.read_csv(test_csv)

    valid_data_dicts = []
    for img, seg in zip(test_df["imgs"], test_df["segs"]):
        if os.path.exists(img) and os.path.exists(seg):  # ✅ Check if files exist
            valid_data_dicts.append({"img": img, "seg": seg})
        else:
            print(f"⚠️ Warning: Missing file {img} or {seg}, skipping...")

    test_ds = Dataset(data=valid_data_dicts, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return test_loader
def test(test_loader, model, directory_name, test_csv_path):
    """
    Performs inference on the test dataset and evaluates the model.

    Args:
        test_loader (DataLoader): DataLoader for test dataset.
        model (torch.nn.Module): Trained model for inference.
        directory_name (str): Directory to save results.
        test_csv_path (str): Path to the test dataset CSV file.

    Returns:
        None
    """
    device = next(model.parameters()).device
    results_dir = os.path.join(directory_name, "test_results_oasis")
    os.makedirs(results_dir, exist_ok=True)

    # ✅ Load test dataset CSV file inside the function
    test_df = pd.read_csv(test_csv_path)

    # ✅ Load best model
    best_model_path = os.path.join(directory_name, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Best model loaded for testing.")
    else:
        print("⚠️ Best model not found. Using current model.")

    model.eval()

    inferer = SlidingWindowInferer(roi_size=(128, 160, 128), sw_batch_size=1, overlap=0.50, mode='gaussian')
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=61)])
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=61)])
    criterion = CrossEntropyLoss()
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)

    total_loss = 0.0
    total_batches = 0
    dice_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img, seg = batch["img"].to(device), batch["seg"].to(device)
            if seg.ndim == 4:
                seg = seg.unsqueeze(1)

            pred_seg = inferer(img, model)
            pred_seg_onehot = [post_pred(i) for i in decollate_batch(pred_seg)]
            seg_onehot = [post_label(i) for i in decollate_batch(seg)]

            # Compute Dice
            dice_metric(y_pred=pred_seg_onehot, y=seg_onehot)
            dice_scores.append(dice_metric.aggregate().item())
            dice_metric.reset()

            # Compute Loss
            loss = criterion(pred_seg, seg.squeeze(1).long())
            total_loss += loss.item()
            total_batches += 1

            # ✅ Load affine matrix correctly
            img_path = test_df.iloc[batch_idx]['imgs']  # Load image path from CSV
            if not os.path.exists(img_path):
                print(f"⚠️ Warning: Missing file {img_path}, skipping...")
                continue  # Skip missing files
            # ✅ Load NIfTI file only once
            original_nifti = nib.load(img_path)
            affine = original_nifti.affine  # ✅ Preserve affine matrix
            header = original_nifti.header  # ✅ Preserve header

            # ✅ Save output as NIfTI file
            save_path = os.path.join(results_dir, os.path.basename(img_path))
            nib.save(nib.Nifti1Image(pred_seg.argmax(dim=1)[0].cpu().numpy().squeeze().astype(np.float32), affine, header), save_path)
            print(f"✅ Test result saved: {save_path}")

    # Compute Metrics
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    std_dice = np.std(dice_scores, ddof=1) if len(dice_scores) > 1 else 0.0

    print(f"📊 Test Results → Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}, Std Dice: {std_dice:.4f}")

    wandb.log({
        "Average Test Loss": avg_loss,
        "Average Dice": avg_dice,
        "Dice Standard Deviation": std_dice
    })

    print("✅ Testing complete.")
def main():
    directory_name = SEG_TEST_OUTPUT_DIR
    os.makedirs(directory_name, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb if not already running
    if wandb.run is None:
        wandb.init(project= SEG_TEST_PROJECT_NAME, settings=wandb.Settings(start_method="fork"))
        wandb.run.name = "test_run"

    # Initialize model
    model = SwinUNETR(
        img_size=(128, 160, 128),
        in_channels=1,
        out_channels=61,
        use_checkpoint=True,
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load best model
    best_model_path = os.path.join(directory_name, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Best model loaded for testing.")
    else:
        print("⚠️ Best model not found. Using current model.")

    # ✅ Define test dataset path correctly
    test_csv_path = OASIS2_TEST_CSV 
    test_loader = load_test_data(test_csv_path, test_transforms)

    # ✅ Pass test_csv_path to test function
    test(test_loader, model, directory_name, test_csv_path)

if __name__ == "__main__":
    main()
