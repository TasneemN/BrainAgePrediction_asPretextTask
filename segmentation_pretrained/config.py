"""
config.py

Central configuration file for segmentation training scripts.

- SEG_PRETRAINED_DATA_CSV: Path to the full data CSV for segmentation training, validation and testing.
- SEG_PRETRAINED_OUTPUT_DIR: Output directory for results and checkpoints.
- SEG_PRETRAINED_PROJECT_NAME: Project name for Weights & Biases (wandb); kept in sync with output directory.
- TRAIN_SAMPLES_PER_SEX: Number of training samples per sex (total training samples will be 2x this value).

Note:
- The pretrained weights (e.g., best_model.pth) must exist in SEG_PRETRAINED_OUTPUT_DIR for transfer learning.
- Update these values here to propagate changes across all scripts that import this config.
"""
import torch

# Path to the full data CSV for segmentation training
SEG_PRETRAINED_DATA_CSV = "/work/souza_lab/tasneem/secondaimdataset/matched_files.csv"

# Output directory for results/checkpoints and project name for wandb
# NOTE: The pretrained weights (e.g., best_model.pth) must exist in `directory_name`
SEG_PRETRAINED_OUTPUT_DIR = "segmentationTinpainting_52SSS"
SEG_PRETRAINED_PROJECT_NAME = SEG_PRETRAINED_OUTPUT_DIR  # Keep project name and directory name in sync


# Number of training samples per sex (total samples will be 2x this value)
TRAIN_SAMPLES_PER_SEX = 26  # Adjust this value as needed for your dataset


# Optimizer and scheduler settings
OPTIMIZER_CLASS = torch.optim.Adam
OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-4}
SCHEDULER_CLASS = torch.optim.lr_scheduler.StepLR
SCHEDULER_PARAMS = {'step_size': 70, 'gamma': 0.6}
MAX_EPOCHS = 200

# Data loading and splitting parameters
TEST_SAMPLES_PER_GROUP = 2 # Number of test samples per group (e.g., age group)
VAL_SAMPLES_PER_GROUP = 2 # Number of validation samples per group (e.g., age group)
TRAIN_SAMPLES_PER_SEX = 39  # or whatever value you use
CACHE_RATE = 1.0
NUM_WORKERS = 4
BATCH_SIZE = 2
SELECTED_AGE_GROUPS = None  # or a list of age groups if you want to specify

# For testing only: you can run test2.py separately.
# This section is for testing the model on the external test set OASIS2.

# Output directory for results/checkpoints and project name for wandb (testing)
SEG_TEST_OUTPUT_DIR = "segmentationTaging_52SSSS"
SEG_TEST_PROJECT_NAME = SEG_TEST_OUTPUT_DIR  # Keep project name and directory name in sync

# Path to the external test CSV (OASIS2)
OASIS2_TEST_CSV = "/work/souza_lab/tasneem/OASIS2_resized/synthstrip/matched_files.csv"