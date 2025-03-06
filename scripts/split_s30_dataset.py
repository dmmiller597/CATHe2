#!/usr/bin/env python3
"""
Script to process the S30 dataset by filtering and splitting into train/evaluation sets.

This script:
1. Loads sequences from s30 dataset where s30_rep=True
2. Keeps only sequence, sequence_id, and H_group (renamed as SF)
3. Filters out SF groups with fewer than 10 sequences
4. Creates a 90/10 train/eval split for each SF group
"""

import os
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process S30 dataset and create train/val/test splits.')
    parser.add_argument('--input', type=str, 
                        default='data/TED/s30/all_cathe2_s90_reps.parquet',
                        help='Path to input parquet file')
    parser.add_argument('--output-dir', type=str, 
                        default='data/TED/s30/processed',
                        help='Directory to save output files')
    parser.add_argument('--min-sequences', type=int, 
                        default=10,
                        help='Minimum number of sequences per SF group')
    parser.add_argument('--val-fraction', type=float, 
                        default=0.05,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test-fraction', type=float, 
                        default=0.05,
                        help='Fraction of data to use for testing')
    parser.add_argument('--seed', type=int, 
                        default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    """Main function to process the dataset."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Load the dataset
    logger.info(f"Loading dataset from {args.input}")
    try:
        df = pd.read_parquet(args.input)
        logger.info(f"Loaded dataset with {len(df):,} rows")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Filter for s30_rep = True
    logger.info("Filtering for s30_rep = True")
    s30_df = df[df.s30_rep == True].copy()
    logger.info(f"Found {len(s30_df):,} sequences with s30_rep = True")
    
    # Select and rename columns
    logger.info("Selecting required columns and renaming H_group to SF")
    s30_df = s30_df[['sequence_id', 'sequence', 'H_group']].rename(columns={'H_group': 'SF'})
    
    # Group by SF and filter groups with < min_sequences
    # Ensure we have at least 3 sequences per SF (minimum for each split)
    actual_min = max(args.min_sequences, 3)
    logger.info(f"Filtering SF groups with fewer than {actual_min} sequences")
    sf_counts = s30_df['SF'].value_counts()
    valid_sfs = sf_counts[sf_counts >= actual_min].index
    s30_df = s30_df[s30_df['SF'].isin(valid_sfs)]
    
    logger.info(f"Retained {len(s30_df):,} sequences from {len(valid_sfs):,} SF groups")
    
    # Calculate fractions ensuring minimum representation
    val_fraction = args.val_fraction
    test_fraction = args.test_fraction
    train_fraction = 1.0 - val_fraction - test_fraction
    
    # Split each group into train/val/test
    logger.info(f"Splitting each SF group: {val_fraction:.0%} val, {test_fraction:.0%} test, {train_fraction:.0%} train")
    logger.info(f"Ensuring all SF groups appear in all three splits")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for sf, group in tqdm(s30_df.groupby('SF'), desc="Processing SF groups"):
        # Shuffle the group
        shuffled = group.sample(frac=1.0)
        group_size = len(shuffled)
        
        # Ensure at least one sequence in each split for this SF
        min_per_split = 1
        
        # Calculate remaining sequences to distribute according to ratios
        remaining = group_size - (3 * min_per_split)
        
        # Calculate additional sequences for each split based on ratios
        if remaining > 0:
            additional_val = int(remaining * val_fraction)
            additional_test = int(remaining * test_fraction)
            additional_train = remaining - additional_val - additional_test
        else:
            additional_train = additional_val = additional_test = 0
        
        # Total sequences for each split
        val_size = min_per_split + additional_val
        test_size = min_per_split + additional_test
        train_size = min_per_split + additional_train
        
        # Double-check that we're using all sequences
        assert val_size + test_size + train_size == group_size, f"Split sizes don't add up: {val_size}+{test_size}+{train_size}!={group_size}"
        
        # Get indices for each split
        current_idx = 0
        
        # Use the first sequences for each split to ensure representation
        val_indices = list(range(current_idx, current_idx + min_per_split))
        current_idx += min_per_split
        
        test_indices = list(range(current_idx, current_idx + min_per_split))
        current_idx += min_per_split
        
        train_indices = list(range(current_idx, current_idx + min_per_split))
        current_idx += min_per_split
        
        # Distribute remaining sequences
        # Additional for val
        val_indices.extend(range(current_idx, current_idx + additional_val))
        current_idx += additional_val
        
        # Additional for test
        test_indices.extend(range(current_idx, current_idx + additional_test))
        current_idx += additional_test
        
        # Additional for train
        train_indices.extend(range(current_idx, current_idx + additional_train))
        
        # Split the data
        train_df = shuffled.iloc[train_indices]
        val_df = shuffled.iloc[val_indices]
        test_df = shuffled.iloc[test_indices]
        
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)
    
    # Combine results
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine everything for a full dataset with split labels
    full_df = pd.concat([train_df, val_df, test_df])
    
    # Save the results
    logger.info(f"Saving train set with {len(train_df):,} sequences")
    train_output = output_dir / 's30_train.parquet'
    train_df.to_parquet(train_output)
    
    logger.info(f"Saving validation set with {len(val_df):,} sequences")
    val_output = output_dir / 's30_val.parquet'
    val_df.to_parquet(val_output)
    
    logger.info(f"Saving test set with {len(test_df):,} sequences")
    test_output = output_dir / 's30_test.parquet'
    test_df.to_parquet(test_output)
    
    logger.info(f"Saving full dataset with {len(full_df):,} sequences")
    full_output = output_dir / 's30_full.parquet'
    full_df.to_parquet(full_output)
    
    # Verify that all SF groups appear in all splits
    train_sfs = set(train_df['SF'].unique())
    val_sfs = set(val_df['SF'].unique())
    test_sfs = set(test_df['SF'].unique())
    
    logger.info(f"SF groups in train: {len(train_sfs)}")
    logger.info(f"SF groups in val: {len(val_sfs)}")
    logger.info(f"SF groups in test: {len(test_sfs)}")
    
    if train_sfs == val_sfs == test_sfs:
        logger.info("✅ All splits contain the same SF groups")
    else:
        logger.warning("⚠️ Splits contain different SF groups - this should not happen")
    
    # Log summary statistics
    logger.info("Dataset processing complete. Summary:")
    logger.info(f"  - Total SF groups: {len(valid_sfs):,}")
    logger.info(f"  - Total sequences: {len(full_df):,}")
    logger.info(f"  - Training sequences: {len(train_df):,} ({len(train_df)/len(full_df):.1%})")
    logger.info(f"  - Validation sequences: {len(val_df):,} ({len(val_df)/len(full_df):.1%})")
    logger.info(f"  - Test sequences: {len(test_df):,} ({len(test_df)/len(full_df):.1%})")
    
    # Log SF group distribution
    sf_dist = full_df.groupby(['SF', 'split']).size().unstack(fill_value=0)
    logger.info(f"SF group statistics:")
    logger.info(f"  - Average train sequences per SF: {sf_dist['train'].mean():.1f}")
    logger.info(f"  - Average val sequences per SF: {sf_dist['val'].mean():.1f}")
    logger.info(f"  - Average test sequences per SF: {sf_dist['test'].mean():.1f}")
    logger.info(f"  - Min train sequences per SF: {sf_dist['train'].min()}")
    logger.info(f"  - Min val sequences per SF: {sf_dist['val'].min()}")
    logger.info(f"  - Min test sequences per SF: {sf_dist['test'].min()}")

if __name__ == "__main__":
    main()
