import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

# Load ProtT5 in half-precision
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer, device


def get_embeddings(model, tokenizer, sequences, device, batch_size):
    """Get ProtT5 embeddings for a list of sequences"""
    all_embeddings = []
    
    # Sort sequences by length for more efficient batching
    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    seq_lengths.sort(key=lambda x: x[1], reverse=True)  # Sort by longest first
    sorted_indices = [x[0] for x in seq_lengths]
    sequences = [sequences[i] for i in sorted_indices]
    
    # Pre-process all sequences at once to avoid redundant processing in the loop
    # Replace rare/ambiguous amino acids by X and introduce white-space between all amino acids
    processed_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    
    with torch.inference_mode():  # More efficient than no_grad for inference
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = processed_sequences[i:i + batch_size]
            
            # Tokenize sequences and pad up to the longest sequence in the batch
            ids = tokenizer.batch_encode_plus(batch_sequences, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids'], device=device)
            attention_mask = torch.tensor(ids['attention_mask'], device=device)
            
            # Generate embeddings
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
                
            # Extract per-protein embeddings
            for j, seq_len in enumerate(attention_mask.sum(dim=1)):
                seq_emb = embedding_repr.last_hidden_state[j, 1:seq_len-1]
                per_protein_emb = seq_emb.mean(dim=0).cpu().numpy()
                all_embeddings.append(per_protein_emb)
    
    return np.array(all_embeddings), sorted_indices

def process_split_data(df_split, split_name, output_dir, model, tokenizer, device, batch_size):
    """Process data for a specific split"""
    print(f"\nProcessing {split_name} split with {len(df_split)} sequences...")
    
    # Get embeddings
    sequences = df_split['sequence'].tolist()
    sequence_ids = df_split['sequence_id'].tolist()
    
    embeddings, sorted_indices = get_embeddings(model, tokenizer, sequences, device, batch_size)
    
    # Reorder metadata based on sorted indices
    sorted_sequence_ids = [sequence_ids[i] for i in sorted_indices]
    sorted_sf = df_split['SF'].values[sorted_indices]
    
    # Create final dataframe with all data
    result_df = pd.DataFrame({
        'sequence_id': sorted_sequence_ids,
        'SF': sorted_sf,
        'embedding': list(embeddings)  # Pass directly as list of numpy arrays
    })
    
    # Save as parquet
    parquet_file = output_dir / f'prot_t5_data_{split_name}.parquet'
    table = pa.Table.from_pandas(result_df)
    pq.write_table(table, parquet_file)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved to {parquet_file}")
    print(f"Number of unique superfamilies: {len(np.unique(sorted_sf))}")
    
    # Skip expensive verification step
    # No need to reload what we just saved

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate ProtT5 embeddings for protein sequences')
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
    print("Loading ProtT5 model...")
    model, tokenizer, device = get_T5_model()
    
    # Process each split separately instead of loading entire dataset
    for split in args.splits if args.splits else pd.read_parquet(args.input)['split'].unique():
        print(f"Loading {split} split data...")
        # Only load the specific split we need
        df_split = pd.read_parquet(
            args.input, 
            filters=[('split', '=', split)]
        ).reset_index(drop=True)
        
        if len(df_split) > 0:
            process_split_data(df_split, split, output_dir, model, tokenizer, device, args.batch_size)
        else:
            print(f"Warning: No data found for split '{split}'. Skipping.")

if __name__ == "__main__":
    main()