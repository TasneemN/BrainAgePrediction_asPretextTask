"""
load_data.py

Handles loading, splitting, and preparing the dataset for 3D brain image segmentation from scratch.

Main features:
- Loads the full dataset metadata from a CSV file.
- Extracts and processes relevant columns (ID, chronological age, sex, image and segmentation paths).
- Stratifies the dataset by sex and age group to ensure balanced splits.
- Splits the data into training, validation, and test sets with configurable sample counts per group.
- Saves the resulting splits as CSV files for reproducibility.
- Prepares MONAI CacheDatasets and ThreadDataLoaders for each split.
- Applies the appropriate MONAI transform pipeline to each split.
- Provides a utility to load the last model checkpoint for resuming or transfer learning.

Functions:
    load_data(...):
        Loads and splits the dataset, saves CSVs, and returns CacheDatasets and DataLoaders for training, validation, and testing.

    load_last_model(model, optimizer, scheduler, directory_name, reset_lr=None):
        Loads the last saved model checkpoint to resume training if available, with optional learning rate reset.
"""


import os
import csv
import pandas as pd
import random
import numpy as np
import torch
from collections import defaultdict
from monai.data import CacheDataset, ThreadDataLoader


def load_data(
    output_dir,
    full_data_path,
    test_samples_per_group=2,
    val_samples_per_group=2,
    train_samples_total=100,
    train_transforms=None,
    val_transforms=None,
    test_transforms=None,
    cache_rate=1.0,
    num_workers=4,
    batch_size=1,
    selected_age_groups=None,
):
    """
    Load and split the dataset into training, validation, and test sets, and configure DataLoaders for each.

    Args:
        full_data_path (str): Path to the CSV file containing the full dataset.
        output_dir (str): Directory to save the split CSV files.
        test_samples_per_group (int): Number of samples per sex and age group for the test set.
        val_samples_per_group (int): Number of samples per sex and age group for the validation set.
        train_samples_total (int): Total number of samples for the training set.
        train_transforms (callable): Transformations to apply to the training data.
        val_transforms (callable): Transformations to apply to the validation data.
        test_transforms (callable): Transformations to apply to the test data.
        cache_rate (float): Fraction of data to cache in memory.
        num_workers (int): Number of worker threads for data loading.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: CacheDatasets and DataLoaders for training, validation, and testing splits.
    """
    # Load the dataset from the CSV file
    full_data = pd.read_csv(full_data_path)

    # Debugging: Print columns and head of the dataset
    print(full_data.columns.tolist())
    print(full_data.head())

    # Extract necessary details
    full_data['Prefix'] = full_data['ID']
    full_data['chronological_age'] = full_data['chronological_age']
    full_data['Sex'] = full_data['Sex']
    full_data['SexGroup'] = full_data['Sex'].map({'M': 'Male', 'F': 'Female'})

    # Define age bins and labels
    age_bins = [(18, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45),
                (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80)]
    age_labels = ['18-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50',
                  '50-55', '55-60', '60-65', '65-70', '70-75', '75-80']
    full_data['AgeGroup'] = pd.cut(full_data['chronological_age'], bins=[x[0] for x in age_bins] + [age_bins[-1][1]],
                                   labels=age_labels, right=False)
    full_data['StratifyGroup'] = full_data['Sex'] + "_" + full_data['AgeGroup'].astype(str)

    # Debugging: Check stratification groups
    print("Unique Stratify Groups:", full_data['StratifyGroup'].unique())

    # Group data by Sex and AgeGroup
    grouped_data = defaultdict(list)
    for _, row in full_data.iterrows():
        grouped_data[(row['Sex'], row['AgeGroup'])].append(row.to_dict())

    test_data, val_data, train_data = [], [], []
    used_filenames = set()
    # Add after grouping data:
    selected_age_groups = selected_age_groups or age_labels  # Default: use all age groups

    # Select samples for test, validation, and training sets
    for (sex, age_group), group_data in grouped_data.items():
        if age_group not in selected_age_groups:
            continue  # Skip this age group if not selected
        if len(group_data) >= test_samples_per_group + val_samples_per_group:
            random.seed(42)
            test_samples = random.sample(group_data, test_samples_per_group)
            val_samples = random.sample([g for g in group_data if g not in test_samples], val_samples_per_group)
            remaining_train_data = [g for g in group_data if g not in test_samples and g not in val_samples]
            train_samples_needed = train_samples_total // len(age_labels)
            train_samples = random.sample(remaining_train_data, min(len(remaining_train_data), train_samples_needed))

            test_data.extend(test_samples)
            val_data.extend(val_samples)
            train_data.extend(train_samples)
        else:
            print(f"Not enough data for Sex: {sex}, AgeGroup: {age_group}")
            print(f"Skipping group: Sex: {sex}, AgeGroup: {age_group} due to insufficient data.")


    # Save datasets to CSV files
    os.makedirs(output_dir, exist_ok=True)
    def save_to_csv(data, filename):
        pd.DataFrame(data).to_csv(filename, index=False)
    save_to_csv(test_data, os.path.join(output_dir, "test_set.csv"))
    save_to_csv(val_data, os.path.join(output_dir, "val_set.csv"))
    save_to_csv(train_data, os.path.join(output_dir, "train_set.csv"))

    # Create CacheDataset and DataLoader
    def create_dataset_and_loader(data, transforms):
        dataset = CacheDataset(data=[{"img": d['imgs'], "seg": d['seg']} for d in data],
                               transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
        loader = ThreadDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataset, loader

    ds_train, train_loader = create_dataset_and_loader(train_data, train_transforms)
    ds_val, val_loader = create_dataset_and_loader(val_data, val_transforms)
    ds_test, test_loader = create_dataset_and_loader(test_data, test_transforms)

    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    return ds_train, train_loader, ds_val, val_loader, ds_test, test_loader

def load_last_model(model, optimizer, scheduler, directory_name, reset_lr=None):
    """
    Load the last saved model checkpoint to resume training if available.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to load state.
        directory_name (str): Directory where model checkpoints are stored.
        reset_lr (float, optional): Optionally reset learning rate to a specified value.

    Returns:
        tuple: Updated model, optimizer, scheduler, start_epoch, last_val_loss, and best_val_loss.
    """
    # Load the last model weights
    last_model_path = os.path.join(directory_name, "last_model.pth")
    if os.path.exists(last_model_path):
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(last_model_path, map_location=map_location) 
        #checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        last_val_loss = checkpoint.get('val_loss', float('inf'))  # Get the last val loss from checkpoint if available
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Get the best val loss from checkpoint if available
        print(f"Last model loaded. Resuming training from epoch {start_epoch}")
        
        # Optionally reset the learning rate if a new learning rate is provided
        if reset_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = reset_lr
            print(f"Learning Rate after resetting: {optimizer.param_groups[0]['lr']}")
        
        return model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss
    else:
        print("Last model weights not found. Starting training from scratch.")
        return model, optimizer, scheduler, 1, float('inf'), float('inf')
