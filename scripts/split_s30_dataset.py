#!/usr/bin/env python3
"""Create stratified dataset splits for S30 protein classification.

Key functionality:
- Filters superfamilies by minimum sequence count requirement
- Creates proportional train/val/test splits for balanced class distribution
- Ensures each superfamily appears in all three splits
- Logs detailed statistics about resulting split distribution

Dependencies: pandas, numpy, scikit-learn
"""

import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process S30 dataset and create splits.')
    parser.add_argument('--input', type=str, 
                        default='data/TED/s30/s30_reps.parquet',
                        help='Path to input parquet file')
    parser.add_argument('--output-dir', type=str, 
                        default='data/TED/s30',
                        help='Directory to save output files')
    parser.add_argument('--min-sequences', type=int, 
                        default=20,
                        help='Minimum number of sequences per SF group')
    parser.add_argument('--val-fraction', type=float, 
                        default=0.05,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test-fraction', type=float, 
                        default=0.05,
                        help='Fraction of data to use for testing')
    parser.add_argument('--seed', type=int, 
                        default=42,
                        help='Random seed')
    return parser.parse_args()

def load_and_filter_data(input_path, min_sequences):
    """Load dataset and filter SF groups with sufficient sequences."""
    try:
        # Load dataset and filter for s30_rep = True
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df):,} sequences")
        
        s30_df = df[df.s30_rep == True][['sequence_id', 'sequence', 'length', 'SF']]
        logger.info(f"Found {len(s30_df):,} sequences with s30_rep = True")
        
        # Filter SF groups with fewer than min_sequences
        sf_counts = s30_df['SF'].value_counts()
        valid_sfs = sf_counts[sf_counts >= min_sequences].index
        filtered_df = s30_df[s30_df['SF'].isin(valid_sfs)]
        
        logger.info(f"Retained {len(filtered_df):,} sequences from {len(valid_sfs):,} 
                    SF groups after min_sequences {min_sequences} filtering")
        return filtered_df
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return None

def create_stratified_splits(df, val_frac, test_frac, seed):
    """Create stratified train/val/test splits preserving SF group representation."""
    np.random.seed(seed)
    
    train_dfs, val_dfs, test_dfs = [], [], []
    
    # Process each SF group separately to maintain stratification
    for sf, group_df in df.groupby('SF'):
        # Shuffle the group
        group_df = group_df.sample(frac=1.0, random_state=seed)
        group_size = len(group_df)
        
        # Calculate split sizes
        val_size = max(1, int(group_size * val_frac))
        test_size = max(1, int(group_size * test_frac))
        
        # Split into train, val, and test sets
        val_df = group_df.iloc[:val_size].copy()
        test_df = group_df.iloc[val_size:val_size+test_size].copy()
        train_df = group_df.iloc[val_size+test_size:].copy()
        
        # Add to collections
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)
    
    # Combine and add split labels
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir):
    """Save the splits to parquet files."""
    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual splits
    train_df.to_parquet(output_dir / 's30_train.parquet')
    val_df.to_parquet(output_dir / 's30_val.parquet')
    test_df.to_parquet(output_dir / 's30_test.parquet')
    
    # Create and save combined dataset
    full_df = pd.concat([train_df, val_df, test_df])
    full_df.to_parquet(output_dir / 's30_full.parquet')
    
    logger.info(f"Saved splits: train ({len(train_df):,}), val ({len(val_df):,}), test ({len(test_df):,})")
    return full_df

def log_statistics(train_df, val_df, test_df, full_df):
    """Log dataset statistics."""
    logger.info(f"Dataset summary:")
    logger.info(f"  - Total SF groups: {len(full_df['SF'].unique()):,}")
    logger.info(f"  - Total sequences: {len(full_df):,}")
    logger.info(f"  - Train: {len(train_df):,} ({len(train_df)/len(full_df):.1%})")
    logger.info(f"  - Val: {len(val_df):,} ({len(val_df)/len(full_df):.1%})")
    logger.info(f"  - Test: {len(test_df):,} ({len(test_df)/len(full_df):.1%})")
    
    # Verify SF groups in splits
    train_sfs = set(train_df['SF'].unique())
    val_sfs = set(val_df['SF'].unique())
    test_sfs = set(test_df['SF'].unique())
    
    if train_sfs == val_sfs == test_sfs:
        logger.info(f"✅ All {len(train_sfs)} SF groups present in all splits")
    else:
        logger.warning("⚠️ SF groups differ between splits")

def main():
    """Main function to process the dataset."""
    args = parse_args()
    
    # Load and filter data
    filtered_df = load_and_filter_data(args.input, args.min_sequences)
    if filtered_df is None:
        return
    
    # Create splits
    logger.info(f"Creating splits: {args.val_fraction:.0%} val, {args.test_fraction:.0%} test, "
                f"{1-args.val_fraction-args.test_fraction:.0%} train")
    train_df, val_df, test_df = create_stratified_splits(
        filtered_df, args.val_fraction, args.test_fraction, args.seed
    )
    
    # Save splits and get combined dataset
    full_df = save_splits(train_df, val_df, test_df, args.output_dir)
    
    # Log statistics
    log_statistics(train_df, val_df, test_df, full_df)

if __name__ == "__main__":
    main()
