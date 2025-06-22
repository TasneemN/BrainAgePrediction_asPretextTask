# config.py

import torch
# Data paths
FULL_DATA_CSV = "/work/souza_lab/tasneem/perfectdataset/matched_files.csv"
TEST_CSV = "/work/souza_lab/tasneem/perfectdataset/test.csv"

#for main_swinunetr.py
# Directory to save models and logs
SWINUNETR_INPAINTING_DIR = "inpainting_model70_3"
SWINUNETR_INPAINTING_PROJECT_NAME = SWINUNETR_INPAINTING_DIR
SWINUNETR_INPAINTING_OUTPUT_DIR = "/work/souza_lab/tasneem/perfectdataset/"
# Optimizer and scheduler settings
SWINUNETR_INPAINTING_OPTIMIZER_CLASS = torch.optim.Adam
SWINUNETR_INPAINTING_OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-4}
SWINUNETR_INPAINTING_SCHEDULER_CLASS = torch.optim.lr_scheduler.StepLR
SWINUNETR_INPAINTING_SCHEDULER_PARAMS = {'step_size': 70, 'gamma': 0.8}
SWINUNETR_INPAINTING_MAX_EPOCHS = 400