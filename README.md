# Voxel_Brain_Age_Prediction_as_a_Pretext_Task
## Overview
This project provides a full pipeline for:
- Creating voxel-level brain age maps
- Pretraining models for inpainting and voxel-level brain age prediction
- Using these pretrained weights for transfer learning or self-supervised learning in 3D brain image segmentation
- Comparing segmentation performance with and without pretraining

## Workflow

### 1. Create Voxel-Level Brain Age Maps
Before any model training, you must generate voxel-level brain age maps for your dataset.  
Navigate to `creating_brain_age_masks/` and run the scripts in this order:
- `assigningageforeachvoxel.py`: Assigns each voxel the participant's age (with optional noise).
- `introducingnoise.py`: Adds additional random noise to the age maps.
- `onemasksheaders.py`: Generates binary brain masks from the images.
- `creatinggroundtruthageprediction.py`: Combines masks and noisy age maps to create ground truth images for age prediction.

### 2. Pretrain Models

#### Ablation Study: Comparison of Backbone Architectures

We trained and compared three different backbone architectures for voxel-level brain age prediction. The results are summarized below:

| Model        | Test loss (MAE) | R²   |
|--------------|-----------------|------|
| UNET         | 6.15 ± 4.2      | 0.84 |
| UNETR        | 7.17 ± 4.4      | 0.81 |
| SWIN UNETR   | 5.86 ± 4.4      | 0.84 |

SwinUNETR achieved the best performance, so we selected it as our foundation model for subsequent inpainting, transfer learning, and segmentation tasks.

**A. Voxel-Level Brain Age Prediction Pretraining**  
_Folder_: `Voxel_level_BrainAgePrediction_pretraining/`  
_Purpose_: Pretrain a model to predict voxel-level brain age maps.  
_Run_:  
- `main_unet.py`, `main_unetr.py`, or `main_swinunetr.py`: Train and save weights for different architectures (UNet, UNETR, SwinUNETR).  
    We trained and compared these models to determine which architecture performs best for voxel-level brain age prediction. SwinUNETR achieved the highest performance, so we selected it as our foundation model for subsequent tasks.

**B. Inpainting Pretraining**  
_Folder_: `Inpainting_pretraining/`  
_Purpose_: Pretrain a model to reconstruct masked regions of brain images.  
_Run_:  
- `main_swinunetr.py` to train the inpainting model and save weights.


### 3. Segmentation Tasks

**A. Segmentation with Transfer Learning (Pretrained Weights)**  
_Folder_: `segmentation_pretrained/`  
_Purpose_: Use weights from inpainting or brain age prediction pretraining for transfer learning in segmentation.  
_Run_:  
- `swinunetrcrossentropy.py` (or variants) to train and evaluate segmentation with pretrained weights.

**B. Segmentation from Scratch**  
_Folder_: `segmentationfromscratch/`  
_Purpose_: Train segmentation models from scratch (no pretraining).  
_Run_:  
- `swinunetrcrossentropy.py` (or variants) to train and evaluate segmentation without pretraining.

### 4. Compare Results
After training both segmentation approaches (with and without pretraining), compare their performance to evaluate the benefit of transfer/self-supervised learning.

## Notes
- All scripts use MONAI and PyTorch for medical image processing and deep learning.
- Experiment tracking is integrated with Weights & Biases (wandb).
- Data loading, augmentation, and loss functions are modular and can be customized in each folder.
- **Make sure to adjust paths and parameters in each script to match your dataset and environment.**

## Important Note

For each main task in this pipeline (brain age mask creation, inpainting pretraining, voxel-level brain age prediction pretraining, and segmentation), **all file and directory paths are centralized in a `config.py` file within each task's folder**.  
**To adapt the pipeline to your environment or dataset, simply update the relevant paths in the corresponding `config.py` file—no need to modify the main scripts directly.**

## Folder Structure
- `creating_brain_age_masks/`: Scripts for generating voxel-level age maps and masks.
- `Inpainting_pretraining/`: Inpainting model training and utilities.
- `Voxel_level_BrainAgePrediction_pretraining/`: Voxel-level brain age prediction model training and utilities.
- `segmentation_pretrained/`: Segmentation with transfer/self-supervised learning.
- `segmentationfromscratch/`: Segmentation from scratch (no pretraining).

## Getting Started
1. Prepare your dataset and update paths in the scripts.
2. Run the brain age mask creation scripts in `creating_brain_age_masks/`.
3. Pretrain models in `Inpainting_pretraining/` and `Voxel_level_BrainAgePrediction_pretraining/`.
4. Train segmentation models in both `segmentation_pretrained/` and `segmentationfromscratch/`.
5. Compare segmentation results to assess the impact of pretraining.

_For more details, see the documentation at the top of each script._  
_If you use this pipeline, please cite the relevant MONAI and PyTorch resources._
