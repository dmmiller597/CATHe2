import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from faesm.esmc import ESMC

# Load ESM-C model
def get_ESM_model(use_half_precision=True, force_cpu=False):
    device = torch.device('cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = ESMC.from_pretrained("esmc_300m", use_flash_attn=True).to(device)
    
    if use_half_precision and not force_cpu:  # half precision only on GPU
        model = model.half()
    
    model = model.eval()
    
    return model, device

def get_embeddings(model, sequences, device, batch_size):
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
            
            # Process each sequence individually to ensure clean embeddings
            batch_embeddings = []
            for j in range(len(batch_sequences)):
                # Process single sequence to get its embedding
                single_sequence = [batch_sequences[j]]
                tokenized = model.tokenizer(single_sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                input_ids = tokenized["input_ids"].to(device)
                single_output = model(input_ids)
                
                # It appears ESM-C returns sequence-level embeddings directly
                # Instead of just taking first position embedding, let's average across all positions
                # ESM-C might have special tokens at start/end, typically positions 0 and last are special tokens
                # Exclude these from the average to get the actual protein representation
                
                # Average all token embeddings (excluding possible special tokens)
                # For safety, we use positions 1:-1 to exclude potential special tokens
                embedding = single_output.embeddings[1:-1].mean(dim=0).cpu().numpy()
                
                # Debugging for first few sequences
                if i == 0 and j < 3:
                    print(f"\nSequence {j} stats:")
                    print(f"  Original sequence length: {len(batch_sequences[j])}")
                    print(f"  averaged embedding shape: {embedding.shape}")
                    print(f"  Example values: {embedding[:5]}")  # First 5 values
                    print(f"  Embeddings shape: {single_output.embeddings.shape}")
                    print(f"  Embeddings[0] shape: {single_output.embeddings[0].shape}")
                    print(f"  Embeddings[1] shape: {single_output.embeddings[1].shape}")
                    print(f"  sequence_logits shape: {single_output.sequence_logits.shape}")



                
                batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings), sorted_indices

def process_split_data(df_split, split_name, output_dir, model, device, batch_size=64):
    """Process data for a specific split"""
    print(f"\nProcessing {split_name} split with {len(df_split)} sequences...")
    
    # Get embeddings
    sequences = df_split['sequence'].tolist()
    sequence_ids = df_split['sequence_id'].tolist()
    
    embeddings, sorted_indices = get_embeddings(model, sequences, device, batch_size)
    
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
    metadata_file = output_dir / f'esmc_labels_{split_name}.csv'
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
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/esmc',
                        help='Output directory for embeddings (default: data/TED/s30/esmc)')
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
    model, device = get_ESM_model(force_cpu=False)
    
    # Load the full dataset
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Process specific splits if requested, otherwise process all splits
    splits_to_process = args.splits if args.splits else df['split'].unique()
    
    for split in splits_to_process:
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) > 0:
            process_split_data(df_split, split, output_dir, model, device, args.batch_size)
        else:
            print(f"Warning: No data found for split '{split}'. Skipping.")

if __name__ == "__main__":
    main()