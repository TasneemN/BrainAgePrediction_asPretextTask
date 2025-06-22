"""
load_data.py

Handles loading, splitting, and preparing the dataset for inpainting pretraining.

Main features:
- Loads the full dataset metadata from a CSV file.
- Extracts and processes relevant columns (ID, age, sex).
- Stratifies the dataset by age group and sex to ensure balanced splits.
- Splits the data into training, validation, and test sets.
- Saves the resulting splits as CSV files for reproducibility.
- Prepares MONAI CacheDatasets and ThreadDataLoaders for each split.
- Applies the appropriate MONAI transform pipeline to each split.

Functions:
    load_data(directory_name, output_dir, test_set_path):
        Loads and splits the dataset, saves CSVs, and returns CacheDatasets and DataLoaders for training, validation, and testing.
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset, ThreadDataLoader
from transforms import *

def load_data(directory_name, output_dir="/work/souza_lab/tasneem/perfectdataset/", test_set_path="/work/souza_lab/tasneem/perfectdataset/test.csv"):
    """
    Load and split the dataset into training, validation, and test sets, save splits as CSV files,
    and configure DataLoaders for each.

    Args:
        directory_name (str): Directory to save the training set CSV for training compatibility.
        output_dir (str): Directory to save the split CSV files.
        test_set_path (str): Path to save the test CSV file if not already present.

    Returns:
        tuple: CacheDatasets and DataLoaders for training, validation, and testing splits.
    """
    # Load the CSV file containing the full dataset
    full_data = pd.read_csv("/work/souza_lab/tasneem/perfectdataset/matched_files.csv")
    # print(full_data.columns.tolist())  # Print all columns
    # print(full_data.head())  # Check the first few rows
    
    # Extract relevant columns
    full_data['Prefix'] = full_data['ID']  # Assume 'ID' corresponds to the prefix
    full_data['chronological_age'] = full_data['chronological_age']
    full_data['Sex'] = full_data['Sex']
    full_data['SexGroup'] = full_data['Sex'].map({'M': 'Male', 'F': 'Female'})

    # Create stratification column
    bins = [18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    labels = ['18-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80']
    full_data['AgeGroup'] = pd.cut(full_data['chronological_age'], bins=bins, labels=labels, right=False)
    full_data['StratifyGroup'] = full_data['Sex'] + "_" + full_data['AgeGroup'].astype(str)
    
    # Remove rows with NaN in StratifyGroup
    full_data = full_data[~full_data['StratifyGroup'].isnull()]
    print(f"Unique StratifyGroup values:\n{full_data['StratifyGroup'].unique()}")

    # Split data
    train_val_data, test_data = train_test_split(
        full_data, test_size=0.15, stratify=full_data['StratifyGroup'], random_state=42
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=0.2941, stratify=train_val_data['StratifyGroup'], random_state=42
    )

    # Save datasets to CSV files
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(directory_name, exist_ok=True)  # Ensure `directory_name` exists
    def save_to_csv(data, filename):
        pd.DataFrame(data).to_csv(filename, index=False)
    save_to_csv(test_data, os.path.join(output_dir, "test_set.csv"))
    save_to_csv(val_data, os.path.join(output_dir, "val_set.csv"))
    save_to_csv(train_data, os.path.join(directory_name, "train_set.csv"))  # Save train set in `directory_name`

  

    # Prepare data for DataLoader
    def extract_data(data):
        imgs_list = list(data['imgs'])
        groundtruth = list(data['imgs'])  # Use the same images as ground truth
        
        return imgs_list, groundtruth

    # Extract data for splits
    imgs_list_train, groundtruth_train = extract_data(train_data)
    imgs_list_val, groundtruth_val = extract_data(val_data)
    imgs_list_test, groundtruth_test = extract_data(test_data)

    filenames_train = [{"img": x, "groundtruth": y} for (x, y) in zip(imgs_list_train, groundtruth_train)]
    filenames_val = [{"img": x, "groundtruth": y} for (x, y) in zip(imgs_list_val, groundtruth_val)]
    filenames_test = [{"img": x, "groundtruth": y} for (x, y) in zip(imgs_list_test, groundtruth_test)]

    # Create CacheDatasets and DataLoaders
    ds_train = CacheDataset(data=filenames_train, transform=train_transforms, cache_rate=2.0, num_workers=0)
    ds_val = CacheDataset(data=filenames_val, transform=val_transforms, cache_rate=2.0, num_workers=0)
    ds_test = CacheDataset(data=filenames_test, transform=test_transforms, cache_rate=2.0, num_workers=0)

    train_loader = ThreadDataLoader(ds_train, num_workers=0, batch_size=2, shuffle=True, drop_last=True)
    val_loader = ThreadDataLoader(ds_val, num_workers=0, batch_size=2, shuffle=True, drop_last=True)
    test_loader = ThreadDataLoader(ds_test, num_workers=0, batch_size=1, shuffle=False, drop_last=True)

    return ds_train, train_loader, ds_val, val_loader, ds_test, test_loader
