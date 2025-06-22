# config.py

import torch

# Output directory and project name for wandb (kept in sync)
SEG_SCRATCH_OUTPUT_DIR = "segmentationscratch_79S"
SEG_SCRATCH_PROJECT_NAME = SEG_SCRATCH_OUTPUT_DIR

# Path to the full data CSV for segmentation from scratch
SEG_SCRATCH_DATA_CSV = "/work/souza_lab/tasneem/secondaimdataset/matched_files.csv"

# Number of training samples per sex (total = 2x this value)
TRAIN_SAMPLES_PER_SEX = 39

# Data loader and training parameters
BATCH_SIZE = 2
NUM_WORKERS = 4
CACHE_RATE = 1.0

# Optimizer and scheduler settings
OPTIMIZER_CLASS = torch.optim.Adam
OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-4}
SCHEDULER_CLASS = torch.optim.lr_scheduler.StepLR
SCHEDULER_PARAMS = {'step_size': 70, 'gamma': 0.6}
MAX_EPOCHS = 200
# external test using OASIS2 by running test2.py separately.

# ===== EXTERNAL TEST DATASET (OASIS2) CONFIGURATION =====
EXT_TEST_OUTPUT_DIR = SEG_SCRATCH_OUTPUT_DIR
EXT_TEST_PROJECT_NAME = EXT_TEST_OUTPUT_DIR + "oasis2"
EXT_TEST_CSV = "/work/souza_lab/tasneem/OASIS2_resized/synthstrip/matched_files.csv"