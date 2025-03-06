import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate protein embeddings using ProtT5.')
    parser.add_argument('--input-dir', type=str,
                        default='data/TED/s30/processed',
                        help='Directory containing the input parquet files')
    parser.add_argument('--output-dir', type=str,
                        default='data/TED/embeddings',
                        help='Directory to save embeddings and labels')
    parser.add_argument('--batch-size', type=int,
                        default=16,
                        help='Batch size for embedding generation')
    parser.add_argument('--file-prefix', type=str,
                        default='s30',
                        help='Prefix of parquet files (e.g., "s30" for s30_train.parquet)')
    return parser.parse_args()

def load_prot_t5():
    """Load ProtT5 model and tokenizer"""
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_bfd')
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Convert to half precision
    model = model.half()  
    model = model.eval()
    
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
        
        input_ids = ids['input_ids'].to(device)  # Keep as Long
        attention_mask = ids['attention_mask'].to(device)  # Keep as Long
        
        with torch.no_grad():
            embedding = model(input_ids=input_ids,
                           attention_mask=attention_mask)
            
            # Get last hidden states
            last_hidden_states = embedding.last_hidden_state.half()  # Convert to half here
            attention_mask = attention_mask.half()  # Convert to half here
            
            # Vectorized mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings), sorted_indices

def process_split(split_name, model, tokenizer, device, args):
    """Process a single data split"""
    print(f"\nProcessing {split_name} split...")
    
    # Load data using the new file path structure
    input_file = Path(args.input_dir) / f'{args.file_prefix}_{split_name}.parquet'
    print(f"Loading data from {input_file}")
    
    try:
        df = pd.read_parquet(input_file)
        print(f"Loaded {len(df)} sequences")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return
    
    # Get embeddings
    print(f"Generating embeddings...")
    sequences = df['sequence'].tolist()
    embeddings, sorted_indices = get_embeddings(model, tokenizer, sequences, device, args.batch_size)
    
    # Get labels (SF) in the same order as the embeddings
    labels = df['SF'].values[sorted_indices]
    
    # Save embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'prot_t5_embeddings_{split_name}.npz'
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Save labels as CSV
    labels_df = pd.DataFrame({'SF': labels})
    labels_file = output_dir / f'Y_{split_name.capitalize()}_SF.csv'
    labels_df.to_csv(labels_file, index=False)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved labels to {labels_file}")
    print(f"Number of unique superfamilies: {len(np.unique(labels))}")
    
    # Also save sequence IDs for reference
    sequence_ids = df['sequence_id'].values[sorted_indices]
    seq_id_df = pd.DataFrame({'sequence_id': sequence_ids, 'SF': labels})
    seq_id_file = output_dir / f'sequence_ids_{split_name}.csv'
    seq_id_df.to_csv(seq_id_file, index=False)
    print(f"Saved sequence IDs to {seq_id_file}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load model and tokenizer
    print("Loading ProtT5 model...")
    model, tokenizer, device = load_prot_t5()
    print(f"Using device: {device}")
    
    # Process all splits
    for split in ['train', 'val', 'test']:
        process_split(split, model, tokenizer, device, args)
    
    print("\nEmbedding generation complete!")

if __name__ == "__main__":
    main()