#!/usr/bin/env python3
"""
Generate per-residue ESM-2 embeddings and store them in an LMDB database.

This script takes a FASTA file of protein sequences as input, generates
per-residue embeddings using a specified ESM-2 model, and saves them into a
Lightning Memory-Mapped Database (LMDB).

Key functionalities:
- Reads protein sequences from a FASTA file.
- Generates embeddings using ESM-2 models from the transformers library.
- Extracts per-residue embeddings from a specific layer.
- Stores embeddings in an LMDB for efficient access.
"""

import argparse
import logging
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_esm_model(model_name, device):
    """Loads the ESM-2 model and tokenizer."""
    logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    model.eval()
    if torch.cuda.is_available():
        model = model.half() # Use half-precision for faster inference
    logging.info("Model loaded successfully.")
    return model, tokenizer

def read_fasta(fasta_file):
    """Reads a FASTA file and yields sequence ID and sequence."""
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            # Extract CATH ID from header format: cath|4_4_0|1oaiA00/561-619
            header = str(record.id)
            if header.startswith('cath|'):
                # Split by '|' and take the third part, then split by '/' to get just the domain ID
                cath_id = header.split('|')[2].split('/')[0]
                yield cath_id, str(record.seq)
            else:
                yield header, str(record.seq)

def main():
    parser = argparse.ArgumentParser(description="Generate per-residue ESM-2 embeddings and store in LMDB.")
    parser.add_argument("--fasta_file", type=str, default="data/CATH/cath_S100.fasta", help="Path to the input FASTA file.")
    parser.add_argument("--lmdb_path", type=str, default="data/CATH/esm2_t33_650M_UR50D_embeddings_layer32.lmdb", help="Path to the output LMDB directory.")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="ESM model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--layer", type=int, default=32, help="Layer to extract embeddings from (e.g., 32 for esm2_t33).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model, tokenizer = get_esm_model(args.model_name, device)

    # Estimate number of sequences for tqdm
    num_sequences = sum(1 for _ in read_fasta(args.fasta_file))

    # Create LMDB environment
    # The map_size must be large enough to hold the entire dataset.
    # Estimate size: num_sequences * avg_seq_len * embedding_dim * 4 bytes/float
    # 250,000 * 400 * 1280 * 4 = 512 GB. Let's use 1TB to be safe.
    map_size = 1024 * 1024 * 1024 * 1024  # 1 TB
    lmdb_path = Path(args.lmdb_path)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size)

    with env.begin(write=True) as txn:
        sequences_generator = read_fasta(args.fasta_file)
        with torch.inference_mode():
            for i in tqdm(range(0, num_sequences, args.batch_size), desc="Generating embeddings"):
                batch_ids = []
                batch_seqs = []
                try:
                    for _ in range(args.batch_size):
                        seq_id, seq = next(sequences_generator)
                        batch_ids.append(seq_id)
                        batch_seqs.append(seq)
                except StopIteration:
                    pass

                if not batch_seqs:
                    continue
                
                # Sort by length for padding efficiency
                sorted_indices = np.argsort([-len(s) for s in batch_seqs])
                batch_seqs = [batch_seqs[i] for i in sorted_indices]
                batch_ids = [batch_ids[i] for i in sorted_indices]

                inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1022) # max_length must be 1022 for esm
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                hidden_states = outputs.hidden_states[args.layer]

                for j, seq_id in enumerate(batch_ids):
                    # Remove [CLS] and [SEP] tokens
                    seq_len = (inputs['attention_mask'][j] == 1).sum()
                    embedding = hidden_states[j, 1:seq_len-1].cpu().numpy()
                    
                    # Store in LMDB
                    serialized_embedding = pickle.dumps(embedding)
                    txn.put(key=seq_id.encode(), value=serialized_embedding)

    logging.info(f"Embeddings successfully saved to {args.lmdb_path}")

if __name__ == "__main__":
    main() 