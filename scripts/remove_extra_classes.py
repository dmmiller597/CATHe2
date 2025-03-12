#!/usr/bin/env python3
"""
Simple script to identify and remove superfamilies that are not present in all splits.
Specifically removes superfamilies that appear in only one of train/val/test splits.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# File paths - update these to match your actual file locations
TRAIN_CSV = "data/annotations/Y_Train_SF.csv"
VAL_CSV = "data/annotations/Y_Val_SF.csv"
TEST_CSV = "data/annotations/Y_Test_SF.csv"
TRAIN_NPZ = "data/embeddings/SF_Train_ProtT5.npz"

# Output paths
OUTPUT_DIR = "data/filtered"
OUTPUT_NPZ = "SF_Train_ProtT5_filtered.npz"
OUTPUT_CSV = "SF_Train_ProtT5_filtered.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading label files to identify unique superfamilies...")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# Extract unique superfamilies from each split
train_sfs = set(train_df['SF'].unique())
val_sfs = set(val_df['SF'].unique())
test_sfs = set(test_df['SF'].unique())

# Find superfamilies unique to each split
train_unique = train_sfs - (val_sfs | test_sfs)
val_unique = val_sfs - (train_sfs | test_sfs)
test_unique = test_sfs - (train_sfs | val_sfs)

# Print analysis
print("\nSuperfamily Analysis:")
print(f"Total superfamilies in training: {len(train_sfs)}")
print(f"Total superfamilies in validation: {len(val_sfs)}")
print(f"Total superfamilies in testing: {len(test_sfs)}")
print(f"Superfamilies in all three sets: {len(train_sfs & val_sfs & test_sfs)}")
print(f"\nSuperfamilies unique to training: {len(train_unique)} - {sorted(train_unique)}")
print(f"Superfamilies unique to validation: {len(val_unique)} - {sorted(val_unique)}")
print(f"Superfamilies unique to testing: {len(test_unique)} - {sorted(test_unique)}")

# Load the training embeddings and labels to filter
print(f"\nLoading training embeddings from {TRAIN_NPZ}...")
data = np.load(TRAIN_NPZ)
embeddings = data['arr_0']  # Assuming 'arr_0' is the key
print(f"Loaded embeddings with shape {embeddings.shape}")

# Create mask to filter out unique superfamilies from training
mask = ~train_df['SF'].isin(train_unique)
filtered_embeddings = embeddings[mask]
filtered_labels = train_df[mask].reset_index(drop=True)

# Print filtering stats
removed_count = len(embeddings) - len(filtered_embeddings)
print(f"\nRemoving {removed_count} samples ({removed_count/len(embeddings):.2%} of the dataset)")
print(f"Original dataset: {len(embeddings)} samples")
print(f"Filtered dataset: {len(filtered_embeddings)} samples")

# Save filtered data
output_npz_path = os.path.join(OUTPUT_DIR, OUTPUT_NPZ)
output_csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)

print(f"\nSaving filtered embeddings to {output_npz_path}...")
np.savez_compressed(output_npz_path, arr_0=filtered_embeddings)

print(f"Saving filtered labels to {output_csv_path}...")
filtered_labels.to_csv(output_csv_path, index=False)

print("\nDone!")
