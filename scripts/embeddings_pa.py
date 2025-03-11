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

# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer, device


def get_embeddings(model, tokenizer, sequences, device, batch_size=16):
    """Get ProtT5 embeddings for a list of sequences"""
    all_embeddings = []
    
    # Sort sequences by length for more efficient batching
    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    seq_lengths.sort(key=lambda x: x[1])
    sorted_indices = [x[0] for x in seq_lengths]
    sequences = [sequences[i] for i in sorted_indices]
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        
        # Replace U, Z, O, B with X
        batch_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_sequences]
        
        # Tokenize sequences
        ids = tokenizer.batch_encode_plus(batch_sequences, 
                                        add_special_tokens=True, 
                                        padding=True,
                                        return_tensors='pt')
        
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        with torch.no_grad():
            embedding = model(input_ids=input_ids,
                           attention_mask=attention_mask)
            
            # Get last hidden states
            last_hidden_states = embedding.last_hidden_state.half()
            attention_mask = attention_mask.half()
            
            # Vectorized mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            all_embeddings.extend(embeddings)
    
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
    
    # Convert embeddings to list of arrays for storage
    embedding_lists = [emb.tolist() for emb in embeddings]
    
    # Create final dataframe with all data
    result_df = pd.DataFrame({
        'sequence_id': sorted_sequence_ids,
        'SF': sorted_sf,
        'embedding': embedding_lists  # Store entire vector as a single column
    })
    
    # Save as parquet
    parquet_file = output_dir / f'prot_t5_data_{split_name}.parquet'
    result_df.to_parquet(parquet_file, index=False)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved combined data to {parquet_file}")
    print(f"Number of unique superfamilies: {len(np.unique(sorted_sf))}")
    
    # Simple verification
    print("Verifying data alignment...")
    loaded_df = pd.read_parquet(parquet_file)
    loaded_embeddings = np.array(loaded_df['embedding'].tolist())
    print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"Data verification: {'✓ PASSED' if loaded_embeddings.shape == embeddings.shape else '✗ FAILED'}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate ProtT5 embeddings for protein sequences')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/s30_full.parquet',
                        help='Input parquet file containing protein sequences (default: data/TED/s30/s30_full.parquet)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/embeddings',
                        help='Output directory for embeddings (default: data/TED/s30/embeddings)')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='Batch size for embedding generation (default: 16)')
    parser.add_argument('--splits', '-s', nargs='+', choices=['train', 'val', 'test'], 
                        help='Process specific splits (e.g., train val test). If not specified, processes all splits.')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading ProtT5 model...")
    model, tokenizer, device = get_T5_model()
    
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