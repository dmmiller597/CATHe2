#!/usr/bin/env python3
"""
Convert CATH dataset: filter columns and rename H_group to SF
"""

import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_dataset(input_path: str, output_path: str) -> None:
    """
    Process CATH dataset:
    1. Keep only sequence_id, sequence, length columns
    2. Rename H_group to SF
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Load the parquet file
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Get original column count and row count for logging
    original_cols = df.shape[1]
    original_rows = df.shape[0]
    
    # Select and rename columns
    logger.info("Filtering columns and renaming H_group to SF")
    df = df[['sequence_id', 'sequence', 'length', 'H_group']]
    df = df.rename(columns={'H_group': 'SF'})
    
    # Save processed dataset
    logger.info(f"Saving processed dataset to {output_path}")
    df.to_parquet(output_path)
    
    logger.info(f"Processing complete: {original_rows:,} rows, reduced from {original_cols} to {len(df.columns)} columns")

def main():
    """Main entry point with error handling."""
    try:
        input_file = "data/TED/all_cathe2_s90_reps.parquet"
        output_file = "data/TED/all_cathe2_s90_filtered.parquet"
        
        process_dataset(input_file, output_file)
        logger.info("Dataset processing completed successfully")
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
