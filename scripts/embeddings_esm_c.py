import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from faesm.esmc import ESMC
import os

# Load ESM-C model
def get_ESM_model(use_half_precision=True, force_cpu=False):
    device = torch.device('cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    print("Loading ESM-C model...")
    model = ESMC.from_pretrained("esmc_300m", use_flash_attn=True).to(device)
    print(f"Model loaded. Model type: {type(model).__name__}")
    
    if use_half_precision and not force_cpu:  # half precision only on GPU
        model = model.half()
        print("Model converted to half precision")
    
    model = model.eval()
    print("Model set to evaluation mode")
    
    return model, device

def get_embeddings(model, sequences, device, batch_size):
    """Get ESM-C embeddings for a list of sequences"""
    all_embeddings = []
    
    print(f"Processing {len(sequences)} sequences with batch size {batch_size}")
    
    # Sort sequences by length for more efficient batching
    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    seq_lengths.sort(key=lambda x: x[1], reverse=True)  # Sort by longest first
    sorted_indices = [x[0] for x in seq_lengths]
    sequences = [sequences[i] for i in sorted_indices]
    
    print(f"Sequence length range: {len(sequences[-1])} to {len(sequences[0])} amino acids")
    
    # Debug a sample sequence
    debug_idx = 0
    print(f"\nSample sequence (idx {debug_idx}): {sequences[debug_idx][:50]}... (length {len(sequences[debug_idx])})")
    
    with torch.inference_mode():  # More efficient than no_grad for inference
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            
            # Print details for first batch
            if i == 0:
                print(f"Batch {i//batch_size + 1} size: {len(batch_sequences)}")
            
            # Process entire batch at once
            tokenized_batch = model.tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_ids = tokenized_batch["input_ids"].to(device)
            
            # Debug tokenization for first batch
            if i == 0:
                print(f"Tokenized input shape: {input_ids.shape}")
                print(f"First sequence tokens: {input_ids[0][:10]}...")
            
            # Get model outputs for the entire batch
            outputs = model(input_ids)  # ESM-C expects just input_ids

            # Debug model output shape
            if i == 0:
                print(f"Model output embeddings shape: {outputs.embeddings.shape}")
            
            batch_embeddings = []
            
            # Calculate mean embeddings for each sequence in the batch
            for j in range(len(batch_sequences)):
                # Get attention mask for determining sequence length (excluding padding)
                attention_mask = tokenized_batch["attention_mask"][j].to(device)
                seq_len = attention_mask.sum().item()
                
                # The model gives protein embeddings directly - don't try to skip tokens
                # Extract the embedding for this sequence from the batch
                protein_emb = outputs.embeddings[j].cpu().numpy()
                
                # Verify embedding dimension and stats
                if i == 0 and j == 0:
                    print(f"\nProtein embedding dimensions: {protein_emb.shape}")
                    print(f"Embedding stats - Min: {protein_emb.min():.4f}, Max: {protein_emb.max():.4f}, Mean: {protein_emb.mean():.4f}, Std: {protein_emb.std():.4f}")
                    print(f"Original sequence length: {len(batch_sequences[j])}, Token count: {seq_len}")
                
                batch_embeddings.append(protein_emb)
            
            all_embeddings.extend(batch_embeddings)
            
            # Debug stats for a few batches
            if i < 3*batch_size and i % batch_size == 0:
                print(f"Processed batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}, embeddings collected: {len(all_embeddings)}")
    
    print(f"Total embeddings collected: {len(all_embeddings)}")
    if len(all_embeddings) != len(sequences):
        print(f"WARNING: Number of embeddings ({len(all_embeddings)}) doesn't match number of sequences ({len(sequences)})")
    
    return np.array(all_embeddings), sorted_indices

def process_split_data(df_split, split_name, output_dir, model, device, batch_size=64):
    """Process data for a specific split"""
    print(f"\nProcessing {split_name} split with {len(df_split)} sequences...")
    
    # Get embeddings
    sequences = df_split['sequence'].tolist()
    sequence_ids = df_split['sequence_id'].tolist()
    
    # Debug sample sequences
    print(f"Sample sequence IDs: {sequence_ids[:3]}")
    print(f"Sample sequences (first 50 chars): {[seq[:50] + '...' for seq in sequences[:3]]}")
    
    embeddings, sorted_indices = get_embeddings(model, sequences, device, batch_size)
    
    # Reorder metadata based on sorted indices
    sorted_sequence_ids = [sequence_ids[i] for i in sorted_indices]
    sorted_sf = df_split['SF'].values[sorted_indices]
    
    # Debug reordering
    print(f"\nReordering check:")
    print(f"Original first 3 sequence IDs: {sequence_ids[:3]}")
    print(f"Sorted first 3 sequence IDs: {sorted_sequence_ids[:3]}")
    print(f"Corresponding sorted indices: {sorted_indices[:3]}")
    
    # Save embeddings
    output_file = output_dir / f'esmc_embeddings_{split_name}.npz'
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Debug embeddings statistics
    print(f"\nEmbeddings stats:")
    print(f"Shape: {embeddings.shape}")
    print(f"Min: {np.min(embeddings):.4f}, Max: {np.max(embeddings):.4f}")
    print(f"Mean: {np.mean(embeddings):.4f}, Std: {np.std(embeddings):.4f}")
    
    # Check for NaN or Inf values
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print("WARNING: Embeddings contain NaN or Inf values!")
    
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
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable additional debugging output')
    
    args = parser.parse_args()
    
    # Set debug flag for module
    os.environ["DEBUG_MODE"] = "1" if args.debug else "0"
    
    print("\n===== ESM-C Embedding Generation =====")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Batch size: {args.batch_size}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nInitializing model...")
    model, device = get_ESM_model(force_cpu=False)
    
    # Load the full dataset
    print(f"\nLoading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} sequences")
    print(f"Data columns: {df.columns.tolist()}")
    print(f"Available splits: {df['split'].unique().tolist()}")
    
    # Process specific splits if requested, otherwise process all splits
    splits_to_process = args.splits if args.splits else df['split'].unique()
    print(f"Processing splits: {splits_to_process}")
    
    for split in splits_to_process:
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) > 0:
            process_split_data(df_split, split, output_dir, model, device, args.batch_size)
        else:
            print(f"Warning: No data found for split '{split}'. Skipping.")
    
    print("\n===== Processing Complete =====")

if __name__ == "__main__":
    main()