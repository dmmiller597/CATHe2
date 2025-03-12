import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# Load ESM-C model
def get_ESM_model(use_half_precision=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = ESMC.from_pretrained("esmc_300m").to(device)
    
    if use_half_precision:
        client = client.half()  # use half-precision
    
    client = client.eval()  # set model to evaluation mode
    
    return client, device

def get_embeddings(client, sequences, device, batch_size):
    """Get ESM-C embeddings for a list of sequences"""
    all_embeddings = []
    
    # Sort sequences by length for more efficient batching
    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    seq_lengths.sort(key=lambda x: x[1], reverse=True)  # Sort by longest first
    sorted_indices = [x[0] for x in seq_lengths]
    sequences = [sequences[i] for i in sorted_indices]
    
    with torch.inference_mode():  # More efficient than no_grad for inference
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            batch_embeddings = []
            
            for seq in batch_sequences:
                # Create ESMProtein object
                protein = ESMProtein(sequence=seq)
                
                # Encode protein
                protein_tensor = client.encode(protein)
                
                # Get embeddings
                logits_output = client.logits(
                    protein_tensor, 
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                
                # Extract per-protein embedding (mean of token embeddings)
                per_protein_emb = logits_output.embeddings.mean(dim=0).cpu().numpy()
                batch_embeddings.append(per_protein_emb)
            
            all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings), sorted_indices

def process_split_data(df_split, split_name, output_dir, client, device, batch_size=64):
    """Process data for a specific split"""
    print(f"\nProcessing {split_name} split with {len(df_split)} sequences...")
    
    # Get embeddings
    sequences = df_split['sequence'].tolist()
    sequence_ids = df_split['sequence_id'].tolist()
    
    embeddings, sorted_indices = get_embeddings(client, sequences, device, batch_size)
    
    # Reorder metadata based on sorted indices
    sorted_sequence_ids = [sequence_ids[i] for i in sorted_indices]
    sorted_sf = df_split['SF'].values[sorted_indices]
    
    # Save embeddings
    output_file = output_dir / f'esmc_embeddings_{split_name}.npz'
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Save metadata (sequence IDs and labels)
    metadata_df = pd.DataFrame({
        'sequence_id': sorted_sequence_ids,
        'SF': sorted_sf
    })
    metadata_file = output_dir / f'labels_{split_name.capitalize()}_esmc.csv'
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to {output_file}")
    print(f"Saved metadata to {metadata_file}")
    print(f"Number of unique superfamilies: {len(np.unique(sorted_sf))}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate ESM-C embeddings for protein sequences')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/s30_full.parquet',
                        help='Input parquet file containing protein sequences (default: data/TED/s30/s30_full.parquet)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/embeddings',
                        help='Output directory for embeddings (default: data/TED/s30/embeddings)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Batch size for embedding generation (default: 64)')
    parser.add_argument('--splits', '-s', nargs='+', choices=['train', 'val', 'test'], 
                        help='Process specific splits (e.g., train val test). If not specified, processes all splits.')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading ESM-C model...")
    client, device = get_ESM_model()
    
    # Load the full dataset
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Process specific splits if requested, otherwise process all splits
    splits_to_process = args.splits if args.splits else df['split'].unique()
    
    for split in splits_to_process:
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) > 0:
            process_split_data(df_split, split, output_dir, client, device, args.batch_size)
        else:
            print(f"Warning: No data found for split '{split}'. Skipping.")

if __name__ == "__main__":
    main()