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
    parser = argparse.ArgumentParser(description='Process S30 dataset and create train/eval splits.')
    parser.add_argument('--input', type=str, 
                        default='data/TED/s30/all_cathe2_s90_reps.parquet',
                        help='Path to input parquet file')
    parser.add_argument('--output-dir', type=str, 
                        default='data/TED/s30/processed',
                        help='Directory to save output files')
    parser.add_argument('--min-sequences', type=int, 
                        default=10,
                        help='Minimum number of sequences per SF group')
    parser.add_argument('--eval-fraction', type=float, 
                        default=0.1,
                        help='Fraction of data to use for evaluation')
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
    logger.info(f"Filtering SF groups with fewer than {args.min_sequences} sequences")
    sf_counts = s30_df['SF'].value_counts()
    valid_sfs = sf_counts[sf_counts >= args.min_sequences].index
    s30_df = s30_df[s30_df['SF'].isin(valid_sfs)]
    
    logger.info(f"Retained {len(s30_df):,} sequences from {len(valid_sfs):,} SF groups")
    
    # Split each group into train/eval
    logger.info(f"Splitting each SF group: {args.eval_fraction:.0%} eval, {1-args.eval_fraction:.0%} train")
    
    train_dfs = []
    eval_dfs = []
    
    for sf, group in tqdm(s30_df.groupby('SF'), desc="Processing SF groups"):
        # Shuffle the group
        shuffled = group.sample(frac=1.0)
        
        # Calculate split point
        eval_size = max(1, int(len(shuffled) * args.eval_fraction))
        
        # Split the data
        eval_df = shuffled.iloc[:eval_size]
        train_df = shuffled.iloc[eval_size:]
        
        train_dfs.append(train_df)
        eval_dfs.append(eval_df)
    
    # Combine results
    train_df = pd.concat(train_dfs)
    eval_df = pd.concat(eval_dfs)
    
    # Add split column
    train_df['split'] = 'train'
    eval_df['split'] = 'eval'
    
    # Combine everything for a full dataset with split labels
    full_df = pd.concat([train_df, eval_df])
    
    # Save the results
    logger.info(f"Saving train set with {len(train_df):,} sequences")
    train_output = output_dir / 's30_train.parquet'
    train_df.to_parquet(train_output)
    
    logger.info(f"Saving eval set with {len(eval_df):,} sequences")
    eval_output = output_dir / 's30_eval.parquet'
    eval_df.to_parquet(eval_output)
    
    logger.info(f"Saving full dataset with {len(full_df):,} sequences")
    full_output = output_dir / 's30_full.parquet'
    full_df.to_parquet(full_output)
    
    # Log summary statistics
    logger.info("Dataset processing complete. Summary:")
    logger.info(f"  - Total SF groups: {len(valid_sfs):,}")
    logger.info(f"  - Total sequences: {len(full_df):,}")
    logger.info(f"  - Training sequences: {len(train_df):,} ({len(train_df)/len(full_df):.1%})")
    logger.info(f"  - Evaluation sequences: {len(eval_df):,} ({len(eval_df)/len(full_df):.1%})")
    
    # Log SF group distribution
    sf_dist = full_df.groupby(['SF', 'split']).size().unstack()
    logger.info(f"SF group statistics:")
    logger.info(f"  - Average train sequences per SF: {sf_dist['train'].mean():.1f}")
    logger.info(f"  - Average eval sequences per SF: {sf_dist['eval'].mean():.1f}")

if __name__ == "__main__":
    main()
