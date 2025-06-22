"""
loss.py

Defines loss functions and evaluation metrics for 3D inpainting tasks.

Main features:
- Initializes MONAI's PerceptualLoss (AlexNet-based) for 3D perceptual loss computation.
- Initializes PSNRMetric for evaluating image reconstruction quality.
- Provides a robust perceptual_inpainting_loss_function that:
    - Computes perceptual loss between predicted and ground truth images.
    - Handles both training and validation modes (enabling/disabling gradients as needed).
    - Ensures tensors are on the correct device.
    - Handles errors gracefully to avoid training crashes.

Functions:
    perceptual_inpainting_loss_function(pred_img, groundtruth, is_training=True):
        Computes perceptual loss for 3D inpainting, with support for training and validation modes.
"""
import torch
import torch.nn as nn
from monai.losses import PerceptualLoss
from monai.metrics import PSNRMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize PSNRMetric
psnr_metric = PSNRMetric(
    max_val=1.0,         # For normalized images in the range [0, 1]
    reduction="mean",    # Compute the mean PSNR across the batch
    get_not_nans=False   # Only return the PSNR values
)


import torch
from monai.losses import PerceptualLoss

# 🔹 Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#🔹 Initialize Perceptual Loss using AlexNet WITHOUT Pretrained Weights
perceptual_loss_fn = PerceptualLoss(
    spatial_dims=3,        # 3D data support
    network_type="alex",   # Uses AlexNet (default)
    pretrained=True,      # No pretrained weights (random initialization)
    is_fake_3d=True,       # Uses 2.5D approach for 3D perceptual loss
).to(device)

# 🔹 Set Perceptual Model to Evaluation Mode
perceptual_loss_fn.eval()

# 🔹 Disable Gradients for Perceptual Loss Network
for param in perceptual_loss_fn.parameters():
    param.requires_grad = False  # Freeze model

def perceptual_inpainting_loss_function(pred_img, groundtruth, is_training=True):
    """
    Computes Perceptual Loss for 3D inpainting tasks using MONAI's AlexNet-based PerceptualLoss,
   
    Args:
        pred_img (torch.Tensor): Predicted image tensor (B, C, D, H, W)
        groundtruth (torch.Tensor): Ground truth tensor (B, C, D, H, W)
      
        is_training (bool): If True, enables gradients. If False (Validation), disables gradients.

    Returns:
        loss (torch.Tensor): Computed perceptual loss
    """
    try:
        # 🔹 Set Perceptual Loss Model Mode
        if is_training:
            perceptual_loss_fn.train()
            for param in perceptual_loss_fn.parameters():
                param.requires_grad = True  # Enable gradients for training
        else:
            perceptual_loss_fn.eval()
            for param in perceptual_loss_fn.parameters():
                param.requires_grad = False  # Freeze parameters during validation

        # 🔹 Ensure tensors are on the correct device
        pred_img = pred_img.to(device)
        groundtruth = groundtruth.to(device)


        # 🔹 Compute Perceptual Loss on Masked Regions
        loss = perceptual_loss_fn(pred_img, groundtruth)

        # 🔹 Ensure gradients are only enabled during training
        if is_training:
            loss.requires_grad_(True)

        return loss

    except Exception as e:
        print(f"❌ Error in perceptual loss function: {e}")
        return torch.tensor(0.0, requires_grad=is_training, device=device)  # Avoid crash during training
