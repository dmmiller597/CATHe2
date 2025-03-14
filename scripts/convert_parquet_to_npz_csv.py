import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm

def convert_parquet_to_npz_csv(input_dir, output_dir, splits=None):
    """Convert parquet embeddings to npz + csv format"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use default splits if none specified
    if splits is None:
        splits = ['train', 'val', 'test']
    
    # Add progress bar for splits
    for split in tqdm(splits, desc="Processing splits", position=0, leave=True):
        print(f"\nProcessing {split} split...")
        parquet_file = input_dir / f'prot_t5_data_{split}.parquet'
        
        # Check if file exists
        if not parquet_file.exists():
            print(f"Warning: File {parquet_file} not found. Skipping.")
            continue
        
        # Load the parquet file
        df = pd.read_parquet(parquet_file)
        
        # Extract embeddings as numpy array with progress bar
        print("Extracting embeddings...")
        embeddings_list = []
        
        # Configure tqdm to show progress within this split
        with tqdm(total=len(df), desc=f"Processing {split} embeddings", position=1, leave=True) as pbar:
            for i, embedding in enumerate(df['embedding'].values):
                embeddings_list.append(embedding)
                pbar.update(1)
                
        embeddings = np.stack(embeddings_list)
        
        # Extract metadata
        metadata_df = df[['sequence_id', 'SF']]
        
        # Save embeddings as npz
        print(f"Saving embeddings to npz... (this may take several minutes for large datasets)")
        npz_file = output_dir / f'protT5_embeddings_{split}.npz'
        np.savez_compressed(npz_file, embeddings=embeddings)
        print(f"Finished saving embeddings to {npz_file}")
        
        # Save metadata as csv
        print(f"Saving metadata to csv...")
        csv_file = output_dir / f'protT5_labels_{split}.csv'
        metadata_df.to_csv(csv_file, index=False)
        print(f"Finished saving metadata to {csv_file}")
        
        print(f"Saved {split} embeddings shape: {embeddings.shape}")

def main():
    parser = argparse.ArgumentParser(description='Convert parquet embeddings to npz+csv format')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/protT5',
                        help='Input directory containing parquet files (default: data/TED/s30/protT5)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/protT5',
                        help='Output directory for npz/csv files (default: data/TED/s30/protT5)')
    parser.add_argument('--splits', '-s', nargs='+', choices=['train', 'val', 'test'], 
                        help='Process specific splits (e.g., train val test). If not specified, processes all splits.')
    
    args = parser.parse_args()
    
    convert_parquet_to_npz_csv(args.input, args.output, args.splits)

if __name__ == "__main__":
    main()