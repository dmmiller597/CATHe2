# scripts/embeddings_cheap_256.py
#!/usr/bin/env python3
"""Generate CHEAP embeddings for protein sequences.

Key functionality:
- Creates protein embeddings using the CHEAP model (shorten_1_dim_256)
- Averages per-token embeddings for entire proteins using the provided mask
- Saves compressed embeddings with metadata for each dataset split

Dependencies: torch, cheap, pandas, numpy
"""

import os
import argparse
import logging # Keep logging if you plan to use it, though not used in original script
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from cheap.pretrained import CHEAP_shorten_1_dim_256

# Load CHEAP model (shorten_1_dim_256)
def get_CHEAP_model():
    """Loads the CHEAP_shorten_1_dim_256 model and moves it to the appropriate device."""
    model_name = "CHEAP_shorten_1_dim_256"
    print(f"Loading {model_name} model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # The CHEAP model object is the pipeline itself
    pipeline = CHEAP_shorten_1_dim_256()
    pipeline = pipeline.to(device)  # move model to GPU/CPU
    pipeline = pipeline.eval()    # set model to evaluation mode

    print(f"{model_name} model loaded on {device}.")
    return pipeline, device


def get_embeddings(pipeline, sequences, device, batch_size):
    """Get CHEAP fixed-length embeddings for a list of sequences"""
    all_fixed_embeddings_list = []
    
    # Sort sequences by length for potentially more efficient batching
    # The CHEAP pipeline might handle padding well, but this is generally a good practice.
    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    # Sort by longest first; if CHEAP prefers shorter first for some reason, this could be reversed
    seq_lengths.sort(key=lambda x: x[1], reverse=True)  
    sorted_indices = [x[0] for x in seq_lengths]
    # This list contains sequences sorted by length
    sorted_sequences = [sequences[i] for i in sorted_indices] 
    
    with torch.inference_mode():  # More efficient than no_grad for inference
        for i in tqdm(range(0, len(sorted_sequences), batch_size), desc="Generating embeddings"):
            batch_sequences = sorted_sequences[i:i + batch_size]
            
            # Get embeddings from CHEAP pipeline
            # emb shape: (batch_size, padded_sequence_length, feature_dim)
            # mask shape: (batch_size, padded_sequence_length)
            # feature_dim will be 256 for CHEAP_shorten_1_dim_256
            emb, mask = pipeline(batch_sequences) 
            
            emb = emb.to(device) # Ensure on correct device if pipeline doesn't guarantee
            mask = mask.to(device)

            # Perform mean pooling to get fixed-length embeddings
            # Expand mask to broadcast with embeddings: (batch_size, padded_seq_len, 1)
            expanded_mask = mask.unsqueeze(-1).float() # Ensure mask is float for multiplication
            
            # Zero out embeddings for padding tokens
            masked_emb = emb * expanded_mask
            
            # Sum embeddings for actual tokens along the sequence dimension (dim=1)
            # Result shape: (batch_size, embedding_dimension)
            summed_emb = torch.sum(masked_emb, dim=1)
            
            # Count actual tokens per sequence (sum along dim=1)
            # Result shape: (batch_size, 1)
            num_tokens = torch.sum(mask, dim=1, keepdim=True).float()
            
            # Avoid division by zero for sequences that might be entirely padding or very short
            epsilon = 1e-9 
            fixed_length_batch_embeddings = summed_emb / (num_tokens + epsilon)
            
            all_fixed_embeddings_list.append(fixed_length_batch_embeddings.cpu().numpy())
    
    if not all_fixed_embeddings_list:
        return np.array([]), sorted_indices # Handle empty sequence list

    # Concatenate all batch embeddings
    # These embeddings correspond to the sorted_sequences
    all_embeddings_np = np.concatenate(all_fixed_embeddings_list, axis=0)
    
    return all_embeddings_np, sorted_indices

def process_split_data(df_split, split_name, output_dir, pipeline, device, batch_size=64):
    """Process data for a specific split using CHEAP model"""
    print(f"\nProcessing {split_name} split with {len(df_split)} sequences...")
    
    sequences = df_split['sequence'].tolist()
    sequence_ids = df_split['sequence_id'].tolist() # Keep for metadata
    
    # Get embeddings. These will be in the order of sequences sorted by length.
    embeddings, sorted_indices_from_get_embeddings = get_embeddings(pipeline, sequences, device, batch_size)
    
    # Reorder metadata (sequence IDs and labels) to match the order of `embeddings`
    # The `sorted_indices_from_get_embeddings` contains the original indices of the sequences
    # that correspond to the `embeddings` array.
    # So, if embeddings[0] corresponds to original_sequences[sorted_indices_from_get_embeddings[0]],
    # then sorted_sequence_ids[0] should be sequence_ids[sorted_indices_from_get_embeddings[0]].
    
    # This reorders the original metadata according to how `get_embeddings` sorted the sequences.
    reordered_sequence_ids = [sequence_ids[i] for i in sorted_indices_from_get_embeddings]
    # Assuming 'SF' (superfamily) or similar label is present, reorder it as well.
    # Make sure the column name 'SF' exists or adjust as needed.
    if 'SF' in df_split.columns:
        reordered_sf = df_split['SF'].values[sorted_indices_from_get_embeddings]
    else:
        # Handle case where 'SF' might not be present, perhaps fill with NaN or skip
        print("Warning: 'SF' column not found in dataframe. SF metadata will not be saved.")
        reordered_sf = [np.nan] * len(reordered_sequence_ids)


    # Save embeddings
    model_name_str = "cheap_s1_d256" # For CHEAP_shorten_1_dim_256
    output_file = output_dir / f'{model_name_str}_embeddings_{split_name}.npz'
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Save metadata (sequence IDs and labels)
    metadata_file = output_dir / f'{model_name_str}_labels_{split_name}.csv'
    metadata_df_dict = {'sequence_id': reordered_sequence_ids}
    if 'SF' in df_split.columns: # Only add SF if it was present
         metadata_df_dict['SF'] = reordered_sf
    
    metadata_df = pd.DataFrame(metadata_df_dict)
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to {output_file}")
    print(f"Saved metadata to {metadata_file}")
    if 'SF' in df_split.columns and len(np.unique(reordered_sf)) > 0 : # Check if sf is not all NaN
        print(f"Number of unique superfamilies: {len(np.unique([s for s in reordered_sf if pd.notna(s)]))}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate CHEAP_shorten_1_dim_256 embeddings for protein sequences')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/s30_full.parquet',
                        help='Input parquet file containing protein sequences (default: data/TED/s30/s30_full.parquet)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/cheap_s1_d256', # Changed default output dir
                        help='Output directory for embeddings (default: data/TED/s30/cheap_s1_d256)')
    parser.add_argument('--batch-size', '-b', type=int, default=64, # CHEAP might benefit from different batch sizes
                        help='Batch size for embedding generation (default: 64)')
    parser.add_argument('--splits', '-s', nargs='+', choices=['train', 'val', 'test'], 
                        help='Process specific splits (e.g., train val test). If not specified, processes all splits in the input file.')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CHEAP model
    # `tokenizer` is not needed separately for CHEAP as the pipeline handles it.
    pipeline, device = get_CHEAP_model() 
    
    # Load the full dataset
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        print(f"Error loading input parquet file {args.input}: {e}")
        return

    if 'sequence' not in df.columns or 'sequence_id' not in df.columns:
        print("Error: Input DataFrame must contain 'sequence' and 'sequence_id' columns.")
        return
    if 'split' not in df.columns and not args.splits:
        print("Error: 'split' column not found in DataFrame, and no specific splits provided via --splits argument.")
        print("Please ensure your data has a 'split' column (e.g., 'train', 'val', 'test') or specify splits to process.")
        return
    elif 'split' not in df.columns and args.splits:
        # If specific splits are given but no split column, assume the whole dataframe is one split
        # And name it, e.g. 'all'. This requires user to know what they are doing.
        # For simplicity, let's require 'split' column if args.splits is not used for all data.
        # Or, if args.splits is given, we can iterate through those and if df['split'] does not exist,
        # assign the df to the first split name given.
        # For now, sticking to the original logic which expects a 'split' column if args.splits is not specified.
        print("Warning: --splits argument provided, but no 'split' column in the data. Will process the entire dataset for each named split.")
        # This case is a bit ambiguous, better to require 'split' column for clarity or process all as one.
        # Let's adjust to process the whole dataframe if 'split' column is missing but args.splits are named.
        # This implies the user wants to label the output with these split names but use the whole df.
        # For now, the code below assumes 'split' column exists if args.splits is empty.
        pass


    # Determine splits to process
    if args.splits:
        splits_to_process = args.splits
        # If 'split' column doesn't exist, we assume user wants to process the whole df for each named split
        if 'split' not in df.columns:
            print(f"Warning: 'split' column not found. Processing the entire dataset for each specified split: {args.splits}")
            for split_name in splits_to_process:
                print(f"Assigning entire dataset to split '{split_name}' for processing.")
                # df_split is the entire dataframe in this case
                process_split_data(df.copy(), split_name, output_dir, pipeline, device, args.batch_size)
        else: # 'split' column exists
            for split_name in splits_to_process:
                df_split = df[df['split'] == split_name].reset_index(drop=True)
                if len(df_split) > 0:
                    process_split_data(df_split, split_name, output_dir, pipeline, device, args.batch_size)
                else:
                    print(f"Warning: No data found for explicitly requested split '{split_name}'. Skipping.")
    elif 'split' in df.columns: # No specific splits requested, use all unique values from 'split' column
        splits_to_process = df['split'].unique()
        for split_name in splits_to_process:
            df_split = df[df['split'] == split_name].reset_index(drop=True)
            if len(df_split) > 0: # Should always be true if split_name came from df['split'].unique()
                process_split_data(df_split, split_name, output_dir, pipeline, device, args.batch_size)
            # else case not really needed here as split_name is from unique values
    else: # No args.splits and no 'split' column
        print("Error: No 'split' column in data and no specific splits provided. Cannot determine how to split data.")
        print("Please add a 'split' column to your data or use the --splits argument.")
        return


if __name__ == "__main__":
    main()