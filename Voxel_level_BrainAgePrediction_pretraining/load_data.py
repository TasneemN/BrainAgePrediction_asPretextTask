"""
load_data.py

Handles loading, splitting, and preparing the dataset for voxel-level brain age prediction pretraining.

Main features:
- Loads the full dataset metadata from a CSV file.
- Extracts and processes relevant columns (Extracts and processes relevant columns, including participant ID, chronological age, sex, image file paths, voxel-level age maps (noisy for training, non-noisy for validation and test), and brain binary mask).
- Stratifies the dataset by age group and sex to ensure balanced splits.
- Splits the data into training, validation, and test sets.
- Saves the resulting test split as a CSV file for reproducibility.
- Plots and saves the age distribution for each split.
- Prepares MONAI CacheDatasets and ThreadDataLoaders for each split.
- Applies the appropriate MONAI transform pipeline to each split.

Functions:
    load_data(test_set_path):
        Loads and splits the dataset, saves the test CSV, plots distributions, and returns
        CacheDatasets and DataLoaders for training, validation, and testing.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset, ThreadDataLoader
from transforms import *
def load_data(full_data_path,
    test_set_path):
    """
    Load and split the dataset into training, validation, and test sets, and configure DataLoaders for each.

    Args:
        test_set_path (str): Path to save the test CSV file if not already present.

    Returns:
        tuple: CacheDatasets and DataLoaders for training, validation, and testing splits.
    """
    # Load the CSV file containing the full dataset
    full_data = pd.read_csv(full_data_path)
    # Print all columns to ensure "nonnoisyage" is there
    print(full_data.columns.tolist())

    # Check the first few rows to ensure data integrity
    print(full_data.head())
    
    # Extract columns directly from the CSV
    full_data['Prefix'] = full_data['ID']  # Assume 'ID' column corresponds to the prefix
    full_data['chronological_age'] = full_data['chronological_age']  # Chronological age already exists
    full_data['Sex'] = full_data['Sex']  # Sex information already exists

    # Debugging: Verify extracted details
    print("First few rows after loading details from CSV:")
    print(full_data.head())

    # Add the SexGroup column
    full_data['SexGroup'] = full_data['Sex'].map({'M': 'Male', 'F': 'Female'})

    # Define age ranges for stratification based on chronological age
    bins = [18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    labels = ['18-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80']
    # Create age groups for stratification
    full_data['AgeGroup'] = pd.cut(full_data['chronological_age'], bins=bins, labels=labels, right=False)

    # Create a stratification column combining Sex and AgeGroup
    full_data['StratifyGroup'] = full_data['Sex'] + "_" + full_data['AgeGroup'].astype(str)

    # Debugging: Check for NaN values in StratifyGroup
    print("Checking for NaN in StratifyGroup:")
    print(full_data['StratifyGroup'].isnull().sum())

    # Debugging: Check the unique StratifyGroup values
    print(f"Unique StratifyGroup values:\n{full_data['StratifyGroup'].unique()}")
    # Filter out rows where AgeGroup is NaN
    full_data = full_data[~full_data['AgeGroup'].isnull()]
    # Debugging: Check remaining unique StratifyGroup values
    print(f"Unique StratifyGroup values after filtering:\n{full_data['StratifyGroup'].unique()}")

    try:
        # Step 1: Split the data into 85% (train_val_data) and 15% (test_data)
        train_val_data, test_data = train_test_split(
            full_data,
            test_size=0.15,  # 15% for test set
            stratify=full_data['StratifyGroup'],  # Stratify based on the combined column
            random_state=42
        )

        # Step 2: Split the 85% train_val_data into 60% train and 25% validation
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=0.2941,  # This ensures validation set is 25% of the entire dataset
            stratify=train_val_data['StratifyGroup'],  # Stratify again
            random_state=42
        )
    except Exception as e:
               print(f"Error during train_test_split: {e}")
               print("Distribution of StratifyGroup before split:")
               print(full_data['StratifyGroup'].value_counts())
               raise 


    # Save the test set to a file if necessary
    if not os.path.exists(test_set_path):
        test_data.to_csv(test_set_path, index=False)
        print(f"Test set saved to {test_set_path}")

    
    # Function to extract data from the dataframes
    def extract_data(data):
        imgs_list = list(data['imgs'])
        voxel_age = list(data['age'])  # Use voxel-level age for model input
        nonnoisyvoxel_age = list(data['nonnoisyage'])
        mask = list(data['mask'])
        return imgs_list, voxel_age, nonnoisyvoxel_age, mask
    
    # Extract data for each split
    imgs_list_train, voxel_age_train, nonnoisyvoxel_age_train, mask_train = extract_data(train_data)
    imgs_list_val, voxel_age_val, nonnoisyvoxel_age_val, mask_val = extract_data(val_data)
    imgs_list_test, voxel_age_test, nonnoisyvoxel_age_test, mask_test = extract_data(test_data)

    # Create lists of dictionaries for each split, using voxel-level age for model input
    filenames_train = [{"img": x, "age": y, "mask": z} for (x, y, z) in zip(imgs_list_train, voxel_age_train, mask_train)]
    filenames_val = [{"img": x, "nonnoisyage": y, "mask": z} for (x, y, z) in zip(imgs_list_val,  nonnoisyvoxel_age_val, mask_val)]
    filenames_test = [{"img": x, "nonnoisyage": y, "mask": z} for (x, y, z) in zip(imgs_list_test, nonnoisyvoxel_age_test, mask_test)]

    # Create CacheDataset objects
    ds_train = CacheDataset(data=filenames_train, transform=train_transforms, cache_rate=1.0, num_workers=4)
    ds_val = CacheDataset(data=filenames_val, transform=val_transforms, cache_rate=1.0, num_workers=4)
    ds_test = CacheDataset(data=filenames_test, transform=test_transforms, cache_rate=1.0, num_workers=4)

    # Create DataLoader objects
    train_loader = ThreadDataLoader(ds_train, num_workers=3, batch_size=2, shuffle=True)
    val_loader = ThreadDataLoader(ds_val, num_workers=3, batch_size=2, shuffle=True)
    test_loader = ThreadDataLoader(ds_test, num_workers=3, batch_size=1, shuffle=False)

    # Return Datasets and DataLoaders
    return ds_train, train_loader, ds_val, val_loader, ds_test, test_loader
