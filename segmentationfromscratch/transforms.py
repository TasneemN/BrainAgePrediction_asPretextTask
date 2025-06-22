"""
transforms.py

Defines MONAI-based data preprocessing and augmentation pipelines for 3D brain image segmentation from scratch.

Main features:
- Provides Compose pipelines for training, validation, and testing.
- Applies consistent intensity scaling and tensor conversion to all data splits.
- Uses random spatial cropping for data augmentation during training and validation.

Data keys:
- "img": The input brain image.
- "seg": The segmentation mask.

Transforms used:
- LoadImaged: Loads NIfTI images and segmentation masks, ensures channel-first format.
- ScaleIntensityd: Scales intensity values of the input image to [0, 1].
- RandSpatialCropd: Randomly crops a spatial region of interest (augmentation).
- ToTensord: Converts images and segmentation masks to PyTorch tensors.

Pipelines:
- train_transforms: For training data, applies augmentation.
- val_transforms: For validation data, applies augmentation.
- test_transforms: For test data, applies only preprocessing (no augmentation).
"""
from monai.transforms import (
    Compose, ToTensord, LoadImaged, ScaleIntensityd, RandSpatialCropd
)

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),  # Rescale the image to [0, 1]
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "seg"]),
])

val_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),  # # Rescale the image to [0, 1]
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "seg"]),
])

test_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),  # Load mask as well
    
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),  # Rescale the image to [0, 1]
    ToTensord(keys=["img", "seg"]),                              # Convert all to tensors
])

