# config.py

import torch
# Data paths
FULL_DATA_CSV = "/work/souza_lab/tasneem/perfectdataset/matched_files.csv"
TEST_CSV = "/work/souza_lab/tasneem/perfectdataset/test.csv"

#for main_swinunetr.py
# Directory to save models and logs
SWINUNETR_BRAIN_AGE_OUTPUT_DIR = "swinunetr_brain_age5"
SWINUNETR_BRAIN_AGE_PROJECT_NAME = SWINUNETR_BRAIN_AGE_OUTPUT_DIR

# Optimizer and scheduler settings
SWINUNETR_LR = 1e-3
SWINUNETR_WEIGHT_DECAY = 1e-4
SWINUNETR_OPTIMIZER_CLASS = torch.optim.Adam
SWINUNETR_OPTIMIZER_PARAMS = {'lr': SWINUNETR_LR, 'weight_decay': SWINUNETR_WEIGHT_DECAY}
SWINUNETR_SCHEDULER_CLASS = torch.optim.lr_scheduler.StepLR
SWINUNETR_SCHEDULER_PARAMS = {'step_size': 150, 'gamma': 0.6}
SWINUNETR_MAX_EPOCHS = 600

#for main_unet.py
# Directory to save models and logs
UNET_OUTPUT_DIR = "unet_brain_age1"
UNET_PROJECT_NAME = UNET_OUTPUT_DIR

# Optimizer and scheduler settings
UNET_OPTIMIZER_CLASS = torch.optim.Adam
UNET_OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-4}
UNET_SCHEDULER_CLASS = torch.optim.lr_scheduler.StepLR
UNET_SCHEDULER_PARAMS = {'step_size': 70, 'gamma': 0.6}
UNET_MAX_EPOCHS = 600

#for main_unetr.py
# Directory to save models and logs
UNETR_OUTPUT_DIR = "unetr_brain_age1"   
UNETR_PROJECT_NAME = UNETR_OUTPUT_DIR
# Optimizer and scheduler settings
UNETR_LR = 1e-3
UNETR_WEIGHT_DECAY = 1e-4
UNETR_OPTIMIZER_CLASS = torch.optim.Adam
UNETR_OPTIMIZER_PARAMS = {'lr': UNETR_LR, 'weight_decay': UNETR_WEIGHT_DECAY}
UNETR_SCHEDULER_CLASS = torch.optim.lr_scheduler.StepLR
UNETR_SCHEDULER_PARAMS = {'step_size': 70, 'gamma': 0.6}
UNETR_MAX_EPOCHS = 600