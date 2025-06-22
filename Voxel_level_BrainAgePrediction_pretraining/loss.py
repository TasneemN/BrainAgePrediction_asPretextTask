"""
loss.py

Defines the loss function for voxel-level brain age prediction.

Main features:
- Implements voxel_mae, a Mean Absolute Error (MAE) loss function that computes the average absolute difference between predicted and ground truth age maps at the voxel level, focusing only on the masked (brain) regions.
- Handles cases where the mask is empty or invalid to prevent training failures.
- Provides detailed logging for debugging and error handling.

Functions:
    voxel_mae(pred_age, ground_truth_age, mask):
        Calculates the mean absolute error between predicted and ground truth voxel-level age maps, considering only the masked (foreground) regions.
"""

import torch

def voxel_mae(pred_age, ground_truth_age, mask):
    """""
      Calculate the Mean Absolute Error (MAE) for voxel-level predictions while focusing on the masked foreground.

    Args:
        pred_age (torch.Tensor): Predicted age map tensor of shape (2, 1, 128 ,160, 128).
        ground_truth_age (torch.Tensor): Ground truth age map tensor with the same shape as pred_age.
        mask (torch.Tensor): binary brain Mask tensor to indicate the relevant regions for MAE calculation.

    Returns:
        torch.Tensor: Mean of voxel-wise MAE computed only on the masked regions.
    """
    voxel_mae = []
    try:
        for i in range(len(ground_truth_age)):
            try:
                if torch.sum(ground_truth_age[i]) > 0:
                    ground_truth = ground_truth_age[i].clone()  # Clone the ground truth age maps
                    prediction = pred_age[i].clone()

                    # Apply the mask to both prediction and ground truth to focus on the foreground
                    masked_prediction = prediction * mask[i]  # Apply mask to prediction
                    masked_ground_truth = ground_truth * mask[i]  # Apply mask to ground truth

                    # Compute loss only on the masked foreground
                    if torch.sum(mask[i]) > 0:  # Ensure that the mask has non-zero elements
                        print(f"Mask for iteration {i} has non-zero elements.")
                        loss_img = torch.sum(torch.abs(masked_prediction - masked_ground_truth)) / torch.sum(mask[i])
                        voxel_mae.append(loss_img)
                    else:
                        print(f"Skipping iteration {i} due to empty mask.")

            except Exception as e:
                print(f"Error occurred in iteration {i}: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

    # If `voxel_mae` is empty, handle the case
    if len(voxel_mae) == 0:
        print("voxel_mae is empty. Skipping the computation.")
        return torch.tensor(0.0)  # Return a zero tensor as default to prevent failure

    # Stack the tensors and compute the mean
    voxel_mae = torch.stack(voxel_mae, 0)
    loss = torch.mean(voxel_mae)
    return loss
