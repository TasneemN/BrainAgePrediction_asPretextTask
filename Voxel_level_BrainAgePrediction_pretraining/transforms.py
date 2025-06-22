"""
transforms.py

Defines MONAI-based data augmentation and preprocessing pipelines for voxel-level brain age prediction pretraining.

Main features:
- Provides Compose pipelines for training, validation, and testing.
- Applies consistent orientation, intensity scaling, and tensor conversion to all data splits.
- Uses random rotation and spatial cropping for data augmentation during training and validation.

Data keys:
- "img": The input brain image.
- "age": The voxel-level age map (noisy, for training).
- "nonnoisyage": The voxel-level age map without noise (for validation and test).
- "mask": The brain binary mask.

Transforms used:
- LoadImaged: Loads NIfTI images and ensures channel-first format.
- Orientationd: Reorients images to RAS (Right-Anterior-Superior) axes.
- ScaleIntensityd: Scales intensity values to [0, 1] for the input image.
- RandRotate90d: Randomly rotates images, age maps, and masks by 90 degrees (augmentation).
- RandSpatialCropd: Randomly crops a spatial region of interest (augmentation).
- ToTensord: Converts images, age maps, and masks to PyTorch tensors.

Pipelines:
- train_transforms: For training data, uses noisy age maps and applies augmentation.
- val_transforms: For validation data, uses non-noisy age maps and applies augmentation.
- test_transforms: For test data, uses non-noisy age maps and applies no augmentation.
"""
from monai.transforms import (
     ToTensord,
    RandSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    RandRotate90d,
)

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["img", "age","mask"], ensure_channel_first=True),
    Orientationd(keys=["img", "age","mask"], axcodes="RAS"),
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
    RandRotate90d(keys=["img", "age","mask"], prob=0.5),
    RandSpatialCropd(keys=["img", "age","mask"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "age", "mask"]),
])


val_transforms = Compose([
    LoadImaged(keys=["img", "nonnoisyage", "mask"], ensure_channel_first=True),
    Orientationd(keys=["img", "nonnoisyage", "mask"], axcodes="RAS"),
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
    RandRotate90d(keys=["img", "nonnoisyage", "mask"], prob=0.5),
    RandSpatialCropd(keys=["img", "nonnoisyage", "mask"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "nonnoisyage", "mask"]),
])

test_transforms = Compose([
    LoadImaged(keys=["img", "nonnoisyage", "mask"], ensure_channel_first=True),  
    Orientationd(keys=["img", "nonnoisyage", "mask"], axcodes="RAS"),
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
    ToTensord(keys=["img", "nonnoisyage", "mask"]),
])