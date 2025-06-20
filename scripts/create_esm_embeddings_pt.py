#!/usr/bin/env python3
"""
Generate per-residue ESM-2 embeddings and store them as .pt files.

This script takes a FASTA file of protein sequences as input, generates
per-residue embeddings using a specified ESM-2 model, and saves them into
individual .pt files named by their CATH ID. The embeddings are extracted
from the final layer of the model.

Key functionalities:
- Reads protein sequences from a FASTA file.
- Generates embeddings using ESM-2 models from the transformers library.
- Supports both standard ESM-2 and Flash Attention for faster inference.
- Extracts per-residue embeddings from the last layer.
- Stores embeddings in .pt files for easy loading with PyTorch.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_esm_model(model_name, device, use_flash_attention=False):
    """Loads the ESM-2 model and tokenizer."""
    logging.info(f"Loading model: {model_name}")
    
    if use_flash_attention:
        try:
            from faesm.esm import FAEsmForMaskedLM
            logging.info("Using Flash Attention ESM model.")
            model = FAEsmForMaskedLM.from_pretrained(model_name)
            tokenizer = model.tokenizer
        except (ImportError, TypeError) as e:
            logging.warning(f"Could not load Flash Attention model (reason: {e}). Falling back to standard ESM model.")
            use_flash_attention = False
    
    if not use_flash_attention:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

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
    parser = argparse.ArgumentParser(description="Generate per-residue ESM-2 embeddings and store as .pt files.")
    parser.add_argument("--fasta_file", type=str, default="data/CATH/CATH_S100.fasta", help="Path to the input FASTA file.")
    parser.add_argument("--output_dir", type=str, default="data/CATH/esm2_embeddings", help="Path to the output directory for .pt files.")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="ESM model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument('--flash-attention', '-f', action='store_true', help='Use Flash Attention-enabled ESM for faster processing')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model_info = "Flash Attention ESM-2" if args.flash_attention else "ESM-2"
    logging.info(f"Loading {model_info} model...")
    model, tokenizer = get_esm_model(args.model_name, device, use_flash_attention=args.flash_attention)

    # Estimate number of sequences for tqdm
    num_sequences = sum(1 for _ in read_fasta(args.fasta_file))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            # Use the last hidden state, which is supported by both model types.
            # ModelOutput objects are dict-like, so we can use key access.
            last_hidden_state = outputs['last_hidden_state']

            for j, seq_id in enumerate(batch_ids):
                # Remove [CLS] and [SEP] tokens
                seq_len = (inputs['attention_mask'][j] == 1).sum()
                embedding = last_hidden_state[j, 1:seq_len-1].cpu()
                
                # Store in .pt file
                output_file = output_dir / f"{seq_id}.pt"
                torch.save(embedding, output_file)

    logging.info(f"Embeddings successfully saved to {args.output_dir}")

if __name__ == "__main__":
    main() 