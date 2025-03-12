import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
import argparse

# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()
    
    # Enable gradient checkpointing to reduce memory usage
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    # Optimize model for inference speed
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer, device


def get_embeddings(model, tokenizer, sequences, device, batch_size):
    """Get ProtT5 embeddings for a list of sequences with optimized processing"""
    all_embeddings = []
    
    # Sort sequences by length for more efficient batching
    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    seq_lengths.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in seq_lengths]
    sequences = [sequences[i] for i in sorted_indices]
    
    # Pre-process all sequences (already efficient enough)
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    
    # Create dynamic batches based on sequence length
    dynamic_batches = []
    i = 0
    max_tokens = 24000  # Adjust based on available GPU memory
    
    while i < len(sequences):
        current_batch = []
        current_tokens = 0
        
        while i < len(sequences) and current_tokens < max_tokens:
            seq_tokens = len(sequences[i]) + 2  # +2 for special tokens
            if current_tokens + seq_tokens <= max_tokens or len(current_batch) == 0:
                current_batch.append(sequences[i])
                current_tokens += seq_tokens
                i += 1
            else:
                break
                
        dynamic_batches.append(current_batch)
    
    with torch.inference_mode():
        for batch in tqdm(dynamic_batches):
            # Tokenize with padding
            ids = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
            
            input_ids = torch.tensor(ids['input_ids'], device=device)
            attention_mask = torch.tensor(ids['attention_mask'], device=device)
            
            # Generate embeddings
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
                
            # Extract per-protein embeddings efficiently
            batch_embeddings = []
            for j, seq_len in enumerate(attention_mask.sum(dim=1)):
                seq_emb = embedding_repr.last_hidden_state[j, 1:seq_len-1]
                per_protein_emb = seq_emb.mean(dim=0).cpu().numpy()
                batch_embeddings.append(per_protein_emb)
                
            all_embeddings.extend(batch_embeddings)
            
            # Manual cleanup to help with GPU memory
            del input_ids, attention_mask, embedding_repr
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return np.array(all_embeddings), sorted_indices


def process_split_data(df_split, split_name, output_dir, model, tokenizer, device, batch_size=16):
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
    output_file = output_dir / f'prot_t5_embeddings_{split_name}.npz'
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Save metadata (sequence IDs and labels)
    metadata_df = pd.DataFrame({
        'sequence_id': sorted_sequence_ids,
        'SF': sorted_sf
    })
    metadata_file = output_dir / f'Y_{split_name.capitalize()}_SF.csv'
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to {output_file}")
    print(f"Saved metadata to {metadata_file}")
    print(f"Number of unique superfamilies: {len(np.unique(sorted_sf))}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate ProtT5 embeddings for protein sequences')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/s30_full.parquet',
                        help='Input parquet file containing protein sequences (default: data/TED/s30/s30_full.parquet)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/embeddings',
                        help='Output directory for embeddings (default: data/TED/s30/embeddings)')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='Batch size guideline (dynamic batching will be used)')
    parser.add_argument('--splits', '-s', nargs='+', choices=['train', 'val', 'test'], 
                        help='Process specific splits (e.g., train val test). If not specified, processes all splits.')
    parser.add_argument('--chunk-size', type=int, default=0,
                        help='Process data in chunks of this size to reduce memory usage. 0 means no chunking.')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading ProtT5 model...")
    model, tokenizer, device = get_T5_model()
    
    # Load data efficiently
    print(f"Loading data from {args.input}...")
    if args.splits:
        # Only load necessary splits
        df_list = []
        for split in args.splits:
            try:
                # Try to use predicate pushdown if available
                split_df = pd.read_parquet(args.input, filters=[('split', '=', split)])
                df_list.append(split_df)
            except:
                # Fallback to standard loading and filtering
                full_df = pd.read_parquet(args.input)
                df_list.append(full_df[full_df['split'] == split])
        df = pd.concat(df_list)
    else:
        df = pd.read_parquet(args.input)
    
    # Process specific splits
    splits_to_process = args.splits if args.splits else df['split'].unique()
    
    for split in splits_to_process:
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) == 0:
            print(f"Warning: No data found for split '{split}'. Skipping.")
            continue
            
        if args.chunk_size > 0 and len(df_split) > args.chunk_size:
            # Process in chunks to reduce memory usage
            for i in range(0, len(df_split), args.chunk_size):
                chunk = df_split.iloc[i:i+args.chunk_size]
                chunk_name = f"{split}_chunk{i//args.chunk_size}"
                process_split_data(chunk, chunk_name, output_dir, model, tokenizer, device, args.batch_size)
                # Try to clear memory between chunks
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        else:
            process_split_data(df_split, split, output_dir, model, tokenizer, device, args.batch_size)


if __name__ == "__main__":
    main()