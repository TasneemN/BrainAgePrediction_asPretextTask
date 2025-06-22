"""
transforms.py

Defines MONAI-based data augmentation and preprocessing pipelines for inpainting pretraining.
Data keys:
- "img": The masked input image (with missing or occluded regions).
- "groundtruth": The original, unmasked (whole) image used as ground truth.

Includes Compose pipelines for training, validation, and testing, with reproducible random seeds.

Transforms used:
- LoadImaged: Loads NIfTI images and ground truth, ensures channel-first format.
- ScaleIntensityd: Scales intensity values to [0, 1] for both image and ground truth.
- RandRotate90d: Randomly rotates images and ground truth by 90 degrees (augmentation).
- RandSpatialCropd: Randomly crops a spatial region of interest.
- RandCoarseDropoutd: Randomly drops out coarse regions in the image (not ground truth).
- ToTensord: Converts images and ground truth to PyTorch tensors.

Random seeds are set for all stochastic transforms for reproducibility.
"""
from monai.transforms import (
     ToTensord,
    RandSpatialCropd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandRotate90d,
    RandCoarseDropoutd,
)
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, RandRotate90d, 
    RandSpatialCropd, RandCoarseDropoutd, ToTensord
)

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["img", "groundtruth"], ensure_channel_first=True),
    ScaleIntensityd(keys=["img", "groundtruth"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
    RandRotate90d(keys=["img", "groundtruth"], prob=0.5),
    RandSpatialCropd(keys=["img", "groundtruth"], roi_size=(128, 160, 128)),
    RandCoarseDropoutd(keys=["img"], holes=12, spatial_size=(32, 32, 32), fill_value=0, prob=1.0),
    ToTensord(keys=["img", "groundtruth"]),
])

# Set random seeds for reproducibility
train_transforms.transforms[2].set_random_state(seed=42)  # RandRotate90d
train_transforms.transforms[3].set_random_state(seed=43)  # RandSpatialCropd
train_transforms.transforms[4].set_random_state(seed=44)  # RandCoarseDropoutd

val_transforms = Compose([
    LoadImaged(keys=["img", "groundtruth"], ensure_channel_first=True),
    ScaleIntensityd(keys=["img", "groundtruth"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
    RandRotate90d(keys=["img", "groundtruth"], prob=0.5),
    RandSpatialCropd(keys=["img", "groundtruth"], roi_size=(128, 160, 128)),
    RandCoarseDropoutd(keys=["img"], holes=12, spatial_size=(32, 32, 32), fill_value=0, prob=1.0),  # Ensure the transform is always applied),
    ToTensord(keys=["img", "groundtruth"]),
])

# Set random seeds for reproducibility
val_transforms.transforms[2].set_random_state(seed=45)  # RandRotate90d
val_transforms.transforms[3].set_random_state(seed=46)  # RandSpatialCropd
val_transforms.transforms[4].set_random_state(seed=47)  # RandCoarseDropoutd

test_transforms = Compose([
    LoadImaged(keys=["img", "groundtruth"], ensure_channel_first=True),  
    ScaleIntensityd(keys=["img", "groundtruth"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
    RandCoarseDropoutd(keys=["img"], holes=12, spatial_size=(32, 32, 32), fill_value=0, prob=1.0),  # Ensure the transform is always applied
    ToTensord(keys=["img", "groundtruth"]),
])

# Set random seed for reproducibility
test_transforms.transforms[2].set_random_state(seed=48)  # RandCoarseDropoutd
