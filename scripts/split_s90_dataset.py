#!/usr/bin/env python3
"""Create stratified dataset splits for S90 protein classification.

Key functionality:
- Filters superfamilies by a minimum sequence count (default: 3).
- Creates 80%/10%/10% train/val/test splits.
- Guarantees each superfamily appears in all three splits
- Logs detailed statistics about the resulting split distribution for verification.
"""

import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Configure logging for clear output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for dataset splitting."""
    parser = argparse.ArgumentParser(
        description='Process the S90 dataset and create optimal stratified splits.'
    )
    parser.add_argument(
        '--input', type=str,
        default='data/TED/s90_reps.parquet',
        help='Path to the input s90 parquet file. (Default: %(default)s)'
    )
    parser.add_argument(
        '--output-file', type=str,
        default='data/TED/s90/s90_splits.parquet',
        help='Path to save the output parquet file with splits. (Default: %(default)s)'
    )
    parser.add_argument(
        '--min-sequences', type=int,
        default=3,
        help='Minimum number of sequences required per SF group. (Default: %(default)s)'
    )
    parser.add_argument(
        '--val-fraction', type=float,
        default=0.10,
        help='Fraction of data to use for the validation set. (Default: %(default)s)'
    )
    parser.add_argument(
        '--test-fraction', type=float,
        default=0.10,
        help='Fraction of data to use for the test set. (Default: %(default)s)'
    )
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed for reproducibility. (Default: %(default)s)'
    )
    return parser.parse_args()

def load_and_filter_data(input_path: str, min_sequences: int) -> pd.DataFrame | None:
    """Load the S90 dataset and filter SF groups with too few sequences."""
    logger.info(f"Loading dataset from: {input_path}")
    try:
        # To handle large datasets efficiently, only load the columns required for
        # splitting. The 'sequence' column, which is the largest, is intentionally
        # omitted to conserve memory.
        columns_to_load = ['sequence_id', 'SF', 'length']
        logger.info(f"Loading columns {columns_to_load} from dataset...")
        df = pd.read_parquet(input_path, columns=columns_to_load)
        logger.info(f"Loaded metadata for {len(df):,} total sequences.")

        # Count sequences per SF to prepare for filtering
        sf_counts = df['SF'].value_counts()
        initial_sf_count = len(sf_counts)
        logger.info(f"Found {initial_sf_count:,} unique SF groups.")
        
        # Filter SF groups with fewer than min_sequences
        valid_sfs = sf_counts[sf_counts >= min_sequences].index
        filtered_df = df[df['SF'].isin(valid_sfs)].copy()
        
        retained_sf_count = len(valid_sfs)
        retained_seq_count = len(filtered_df)
        
        logger.info(
            f"After filtering for min {min_sequences} sequences, retained "
            f"{retained_seq_count:,} sequences ({retained_seq_count/len(df):.1%}) "
            f"from {retained_sf_count:,} SF groups ({retained_sf_count/initial_sf_count:.1%})."
        )
        return filtered_df
    except FileNotFoundError:
        logger.error(f"Input file not found at: {input_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading or filtering the data: {e}")
        return None

def create_stratified_splits(
    df: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> pd.DataFrame:
    """Create stratified train/val/test splits, ensuring SF representation."""
    np.random.seed(seed)
    
    df_with_splits = df.copy()
    # Default to train, as it's the largest set. Then overwrite val/test.
    df_with_splits['split'] = 'train'
    
    # Process each SF group separately to maintain perfect stratification
    grouped_by_sf = df_with_splits.groupby('SF')
    
    for sf, group_df in tqdm(grouped_by_sf, total=len(grouped_by_sf), desc="Splitting superfamilies"):
        # Shuffle the group to ensure random splits
        shuffled_group = group_df.sample(frac=1.0, random_state=seed)
        group_size = len(shuffled_group)
        
        # Calculate split sizes, ensuring at least one sample in val/test
        val_size = max(1, int(group_size * val_frac))
        test_size = max(1, int(group_size * test_frac))
        
        # With min_sequences>=3 and val/test fractions < 0.5, the train
        # set is guaranteed to be non-empty.
        
        # Get indices for val and test sets
        test_indices = shuffled_group.index[:test_size]
        val_indices = shuffled_group.index[test_size:test_size + val_size]
        
        # Assign splits. The rest remain 'train' by default.
        df_with_splits.loc[test_indices, 'split'] = 'test'
        df_with_splits.loc[val_indices, 'split'] = 'val'

    return df_with_splits

def save_splits(full_df: pd.DataFrame, output_file: str) -> None:
    """Save the combined dataset with a 'split' column to a single parquet file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the combined dataset
    full_df.to_parquet(output_path)
    
    logger.info(f"Saved combined dataset with splits to: {output_path}")
    
    split_counts = full_df['split'].value_counts()
    logger.info(
        f"Split counts: train ({split_counts.get('train', 0):,}), "
        f"val ({split_counts.get('val', 0):,}), "
        f"test ({split_counts.get('test', 0):,})"
    )

def log_statistics(full_df: pd.DataFrame):
    """Log final summary statistics of the dataset splits."""
    logger.info("---------- Dataset Split Summary ----------")
    total_seqs = len(full_df)
    
    # Derive split-specific dataframes for analysis
    train_df = full_df[full_df['split'] == 'train']
    val_df = full_df[full_df['split'] == 'val']
    test_df = full_df[full_df['split'] == 'test']
    
    logger.info(f"Total SF groups: {len(full_df['SF'].unique()):,}")
    logger.info(f"Total sequences: {total_seqs:,}")
    logger.info(f"  - Train: {len(train_df):,} ({len(train_df)/total_seqs:.1%})")
    logger.info(f"  - Val:   {len(val_df):,} ({len(val_df)/total_seqs:.1%})")
    logger.info(f"  - Test:  {len(test_df):,} ({len(test_df)/total_seqs:.1%})")
    
    # Critical check: Verify that all SF groups are present in all splits
    train_sfs = set(train_df['SF'].unique())
    val_sfs = set(val_df['SF'].unique())
    test_sfs = set(test_df['SF'].unique())
    
    if train_sfs == val_sfs == test_sfs:
        logger.info(f"✅ Success: All {len(train_sfs):,} SF groups are present in all three splits.")
    else:
        logger.warning("⚠️ Warning: SF group sets differ between splits. This should not happen.")
        # Provide debug info if the check fails
        if len(train_sfs - val_sfs) > 0:
            logger.warning(f"SFs in train but not val: {train_sfs - val_sfs}")
        if len(val_sfs - train_sfs) > 0:
            logger.warning(f"SFs in val but not train: {val_sfs - train_sfs}")
    logger.info("-------------------------------------------")

def main():
    """Main function to orchestrate the dataset splitting process."""
    args = parse_args()
    
    filtered_df = load_and_filter_data(args.input, args.min_sequences)
    if filtered_df is None or filtered_df.empty:
        logger.error("Stopping due to data loading/filtering errors or empty dataframe.")
        return
        
    train_frac = 1.0 - args.val_fraction - args.test_fraction
    logger.info(
        f"Creating splits with fractions: "
        f"{train_frac:.0%} train, {args.val_fraction:.0%} val, {args.test_fraction:.0%} test"
    )
    full_df_with_splits = create_stratified_splits(
        filtered_df, args.val_fraction, args.test_fraction, args.seed
    )

    # Create integer labels for SFs, essential for training classifiers
    logger.info("Creating integer labels for Superfamilies.")
    sf_labels = sorted(full_df_with_splits['SF'].unique())
    sf_to_int = {sf: i for i, sf in enumerate(sf_labels)}
    full_df_with_splits['SF_label'] = full_df_with_splits['SF'].map(sf_to_int)

    # Save the mapping to a JSON file for later use (e.g., in the model)
    output_path = Path(args.output_file)
    mapping_filename = f"{output_path.stem}_sf_map.json"
    mapping_path = output_path.parent / mapping_filename
    with open(mapping_path, 'w') as f:
        json.dump(sf_to_int, f, indent=4)
    logger.info(f"Saved Superfamily to integer mapping to: {mapping_path}")

    save_splits(full_df_with_splits, args.output_file)
    
    log_statistics(full_df_with_splits)
    
    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()
