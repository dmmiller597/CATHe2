#!/usr/bin/env python3
"""Generate ProtT5 embeddings for protein sequences.

Key functionality:
- Processes protein sequences using ProtT5-XL-U50 encoder
- Averages per-amino-acid embeddings for entire proteins
- Handles rare/ambiguous amino acids through substitution
- Saves compressed embeddings with metadata for each dataset split

Dependencies: torch, transformers, pandas, numpy
"""

import os
import argparse
import logging
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
from typing import List

# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, legacy=True)

    return model, tokenizer, device

# ────────────────────────────────────────────────────────────────────────────────
# Helper: dynamic token batching
# ────────────────────────────────────────────────────────────────────────────────

def _iter_token_batches(seqs: List[str], token_budget: int):
    """Yield lists of sequences whose combined (len+2) ≤ *token_budget*.

    The +2 accounts for the <s> and </s> special tokens that the ProtT5 tokenizer
    adds to every sequence.  Each sequence stays intact – never split across
    batches – so per-sequence pooling remains correct.
    """

    buffer: List[str] = []
    running = 0

    for seq in seqs:
        tokens = len(seq) + 2  # reserve for special tokens
        # flush if adding the next sequence would exceed the budget
        if buffer and running + tokens > token_budget:
            yield buffer
            buffer, running = [], 0

        buffer.append(seq)
        running += tokens

    if buffer:
        yield buffer

def get_embeddings(
    model: T5EncoderModel,
    tokenizer: T5Tokenizer,
    sequences: List[str],
    device: torch.device,
    max_tokens: int,
):
    """Encode *sequences* with a dynamic token budget.

    Returns
    -------
    np.ndarray
        (N, 1024) array with one row per input sequence in *original order*.
    """

    # 1️⃣  length-sort for faster batching, remember indices
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)
    rev_order = np.argsort(order)  # to restore original order later
    seqs_sorted = [sequences[i] for i in order]

    # 2️⃣  lightweight preprocessing (replace rare AAs, add whitespace)
    seqs_sorted = [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in seqs_sorted]

    embeddings: List[np.ndarray] = []
    with torch.inference_mode():
        pbar = tqdm(total=len(seqs_sorted), desc="Embedding", unit="seq")
        for batch in _iter_token_batches(seqs_sorted, max_tokens):
            ids = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids["input_ids"], device=device)
            attention_mask = torch.tensor(ids["attention_mask"], device=device)

            reps = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            # mean-pool per sequence (skip <s> & </s>)
            for j, seq_len in enumerate(attention_mask.sum(dim=1)):
                seq_emb = reps[j, 1:seq_len - 1]
                embeddings.append(seq_emb.mean(dim=0).cpu().numpy())

            pbar.update(len(batch))
        pbar.close()

    emb_arr = np.vstack(embeddings)[rev_order]  # restore original order
    return emb_arr

def process_split_data(
    df_split: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    model: T5EncoderModel,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_tokens: int = 8000,
):
    """Process data for a specific split"""
    print(f"\nProcessing {split_name} split with {len(df_split)} sequences...")
    
    # Get embeddings
    sequences = df_split['sequence'].tolist()
    sequence_ids = df_split['sequence_id'].tolist()
    
    embeddings = get_embeddings(model, tokenizer, sequences, device, max_tokens)
    
    # Save embeddings
    output_file = output_dir / f'protT5_embeddings_{split_name}.npz'
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Metadata kept in original DataFrame order
    metadata_df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'SF': df_split['SF'].values
    })
    metadata_file = output_dir / f'protT5_labels_{split_name}.csv'
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Saved {split_name} embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to {output_file}")
    print(f"Saved metadata to {metadata_file}")
    print(f"Number of unique superfamilies: {len(np.unique(df_split['SF'].values))}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate ProtT5 embeddings for protein sequences')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/s30_full.parquet',
                        help='Input parquet file containing protein sequences (default: data/TED/s30/s30_full.parquet)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/protT5',
                        help='Output directory for embeddings (default: data/TED/s30/protT5)')
    parser.add_argument('--max-tokens', '-t', type=int, default=32000,
                        help='Maximum total tokens per batch (default: 32000)')
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
            process_split_data(df_split, split, output_dir, model, tokenizer, device, args.max_tokens)
        else:
            print(f"Warning: No data found for split '{split}'. Skipping.")

if __name__ == "__main__":
    main()