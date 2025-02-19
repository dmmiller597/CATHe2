import pandas as pd
import torch
from transformers import T5Model, T5Tokenizer
import numpy as np
from tqdm import tqdm
import re

def load_prot_t5():
    """Load ProtT5 model and tokenizer"""
    model = T5Model.from_pretrained('Rostlab/prot_t5_xl_bfd')
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    return model, tokenizer, device

def get_embeddings(model, tokenizer, sequences, device, batch_size=4):
    """Get ProtT5 embeddings for a list of sequences"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        
        # Replace U, Z, O, B with X
        batch_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_sequences]
        
        # Tokenize sequences
        ids = tokenizer.batch_encode_plus(batch_sequences, 
                                        add_special_tokens=True, 
                                        padding=True)
        
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        
        with torch.no_grad():
            embedding = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=None)
            
            # Use encoder embeddings
            encoder_embedding = embedding[2]
            
            # Average pool the embeddings
            embeddings = []
            for j, mask in enumerate(attention_mask):
                seq_embedding = encoder_embedding[j][mask.bool()].mean(dim=0)
                embeddings.append(seq_embedding.cpu().numpy())
            
            all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings)

def process_split(split_name, model, tokenizer, device):
    """Process a single data split"""
    print(f"\nProcessing {split_name} split...")
    
    # Load data
    df = pd.read_parquet(f"data/splits/{split_name}.parquet")
    
    # Filter for s30_rep = True only for train split
    if split_name == 'train':
        df = df[df['s30_rep'] == True].reset_index(drop=True)
    
    # Get embeddings
    print(f"Generating embeddings for {len(df)} sequences...")
    sequences = df['sequence'].tolist()
    embeddings = get_embeddings(model, tokenizer, sequences, device)
    
    # Get labels
    labels = df['SF'].values
    
    # Save results
    output_file = f'prot_t5_embeddings_{split_name}.npz'
    np.savez_compressed(output_file,
                       embeddings=embeddings,
                       labels=labels)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Number of unique superfamilies: {len(np.unique(labels))}")

def main():
    # Load model and tokenizer
    print("Loading ProtT5 model...")
    model, tokenizer, device = load_prot_t5()
    
    # Process all splits
    for split in ['train', 'val', 'test']:
        process_split(split, model, tokenizer, device)

if __name__ == "__main__":
    main()