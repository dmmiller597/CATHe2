import pandas as pd
import numpy as np
from pathlib import Path
import logging

def cap_outlier_sfs(iqr_multiplier=1.5, min_sequences=10, random_seed=42):
    """
    Load validation and test sets, cap the number of sequences per superfamily using 
    an IQR-based outlier detection approach, and save the filtered versions.
    
    Args:
        iqr_multiplier: Multiplier for IQR to determine outlier threshold (default 1.5)
        min_sequences: Minimum number of sequences to keep per SF
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (balanced_val_df, balanced_test_df)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    np.random.seed(random_seed)
    
    # Load datasets with error handling
    data_dir = Path('data/splits')
    try:
        val_df = pd.read_parquet(data_dir / 'val.parquet')
        test_df = pd.read_parquet(data_dir / 'test.parquet')
    except FileNotFoundError as e:
        logger.error(f"Error loading data files: {e}")
        raise
    
    # Validate input data
    for name, df in [("Validation", val_df), ("Test", test_df)]:
        if 'SF' not in df.columns:
            raise ValueError(f"'SF' column missing in {name} dataset")
        if df.empty:
            raise ValueError(f"{name} dataset is empty")
    
    logger.info(f"Loaded datasets - Val: {len(val_df):,}, Test: {len(test_df):,} sequences")
    
    def calculate_cap(df, name):
        sf_sizes = df['SF'].value_counts()
        log_sizes = np.log10(sf_sizes)
        
        q1 = log_sizes.quantile(0.25)
        q3 = log_sizes.quantile(0.75)
        iqr = q3 - q1
        
        upper_bound = q3 + (iqr_multiplier * iqr)
        max_sequences = int(10 ** upper_bound)
        
        return max_sequences
    
    def cap_dataset(df, max_seqs):
        df_copy = df.copy()
        sf_sizes = df_copy['SF'].value_counts()
        total_removed = 0
        
        for sf, size in sf_sizes.items():
            if size > max_seqs:
                sf_mask = df_copy['SF'] == sf
                sf_sequences = df_copy[sf_mask]
                
                keep_size = max(max_seqs, min_sequences)
                keep_idx = np.random.choice(sf_sequences.index, keep_size, replace=False)
                df_copy = df_copy.loc[df_copy.index.isin(keep_idx) | ~sf_mask]
                
                total_removed += size - keep_size
        
        return df_copy, total_removed
    
    # Calculate and apply caps
    val_max_seqs = calculate_cap(val_df, "Validation")
    test_max_seqs = calculate_cap(test_df, "Test")
    
    logger.info(f"\nCalculated caps - Val: {val_max_seqs:,}, Test: {test_max_seqs:,} sequences per SF")
    
    balanced_val_df, val_removed = cap_dataset(val_df, val_max_seqs)
    balanced_test_df, test_removed = cap_dataset(test_df, test_max_seqs)
    
    # Print statistics
    logger.info("\nSequences per split:")
    logger.info(f"Validation: {len(val_df):,} → {len(balanced_val_df):,} ({val_removed:,} removed)")
    logger.info(f"Test: {len(test_df):,} → {len(balanced_test_df):,} ({test_removed:,} removed)")
    
    
    try:
        balanced_val_df.to_parquet(data_dir / 'val_filtered.parquet')
        balanced_test_df.to_parquet(data_dir / 'test_filtered.parquet')
        logger.info(f"\nSaved filtered datasets to {data_dir}")
    except Exception as e:
        logger.error(f"Error saving filtered datasets: {e}")
        raise
    
    return balanced_val_df, balanced_test_df

if __name__ == "__main__":
    cap_outlier_sfs()