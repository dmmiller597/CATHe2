import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

# Load ESM-2 model (650M parameters)
def get_ESM_model(use_half_precision=True):
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)  # move model to GPU
    if use_half_precision:
        model = model.half()  # use half-precision
    model = model.eval()  # set model to evaluation mode

    return model, tokenizer, device


def get_embeddings(model, tokenizer, sequences, device, batch_size):
    """Get ESM-2 embeddings for a list of sequences"""
    all_embeddings = []
    
    # Sort sequences by length for more efficient batching
    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    seq_lengths.sort(key=lambda x: x[1], reverse=True)  # Sort by longest first
    sorted_indices = [x[0] for x in seq_lengths]
    sequences = [sequences[i] for i in sorted_indices]
    
    with torch.inference_mode():  # More efficient than no_grad for inference
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            
            # ESM-2 expects raw sequences
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate embeddings
            outputs = model(**inputs)
            
            # Extract per-protein embeddings (mean of token embeddings except cls/eos tokens)
            for j, (seq, attn_mask) in enumerate(zip(batch_sequences, inputs['attention_mask'])):
                # Get the representation excluding the start/end tokens
                seq_len = attn_mask.sum().item()
                # For ESM-2, skip the first token (BOS/CLS) and include up to the last token before padding
                # We also exclude the EOS token at the end
                seq_emb = outputs.last_hidden_state[j, 1:seq_len-1]
                # Mean of all token embeddings for the protein
                per_protein_emb = seq_emb.mean(dim=0).cpu().numpy()
                all_embeddings.append(per_protein_emb)
    
    return np.array(all_embeddings), sorted_indices

def process_split_data(df_split, split_name, output_dir, model, tokenizer, device, batch_size=64):
    """Process data for a specific split"""
    print(f"\nProcessing {split_name} split with {len(df_split)} sequences...")
    
    # Get embeddings
    sequences = df_split['sequence'].tolist()
    sequence_ids = df_split['sequence_id'].tolist()
    
    embeddings, sorted_indices = get_embeddings(model, tokenizer, sequences, device, batch_size)
    
    # Reorder metadata based on sorted indices
    sorted_sequence_ids = [sequence_ids[i] for i in sorted_indices]
    sorted_sf = df_split['SF'].values[sorted_indices]
    
    # Save embeddings
    output_file = output_dir / f'esm2_embeddings_{split_name}.npz'
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Save metadata (sequence IDs and labels)
    metadata_df = pd.DataFrame({
        'sequence_id': sorted_sequence_ids,
        'SF': sorted_sf
    })
    metadata_file = output_dir / f'labels_{split_name.capitalize()}_esm.csv'
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to {output_file}")
    print(f"Saved metadata to {metadata_file}")
    print(f"Number of unique superfamilies: {len(np.unique(sorted_sf))}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate ESM-2 embeddings for protein sequences')
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
    
    # Load model and tokenizer
    print("Loading ESM-2 model (650M parameters)...")
    model, tokenizer, device = get_ESM_model()
    
    # Load the full dataset
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Process specific splits if requested, otherwise process all splits
    splits_to_process = args.splits if args.splits else df['split'].unique()
    
    for split in splits_to_process:
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) > 0:
            process_split_data(df_split, split, output_dir, model, tokenizer, device, args.batch_size)
        else:
            print(f"Warning: No data found for split '{split}'. Skipping.")

if __name__ == "__main__":
    main()