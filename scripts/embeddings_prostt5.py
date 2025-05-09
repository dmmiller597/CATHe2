#!/usr/bin/env python3
"""Generate ProstT5 embeddings for protein sequences (AA and 3Di from AA).

Key functionality:
- Processes protein sequences using ProstT5 encoder for AA embeddings.
- Translates AA sequences to 3Di sequences using ProstT5 seq2seq model.
- Generates embeddings from these translated 3Di sequences.
- Averages per-residue embeddings for entire proteins.
- Handles rare/ambiguous amino acids and applies ProstT5-specific prefixes.
- Saves compressed embeddings with metadata for each dataset split,
  separately for AA and 3Di-from-AA embeddings.

Dependencies: torch, transformers, pandas, numpy, tqdm, re
"""

import os
import argparse
import logging
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoModelForSeq2SeqLM
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_prostT5_models_and_tokenizer():
    """Load ProstT5 models (encoder and seq2seq) and tokenizer."""
    logging.info("Loading ProstT5 models and tokenizer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model_name = "Rostlab/ProstT5"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    
    encoder_model = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder_model.eval()

    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    translation_model.eval()

    if device.type == 'cuda':
        logging.info("Using half precision for models on GPU.")
        encoder_model.half()
        translation_model.half()
    else:
        logging.info("Using full precision for models on CPU (slower).")
        # .full() is not a standard method, ensure models are float32 if not on GPU
        encoder_model.float()
        translation_model.float()
        
    logging.info("ProstT5 models and tokenizer loaded successfully.")
    return encoder_model, translation_model, tokenizer, device


def preprocess_sequences_for_prostt5(original_sequences, sequence_type):
    """Preprocess sequences for ProstT5 embedding or translation."""
    processed_sequences = []
    for seq in original_sequences:
        if sequence_type == 'aa':
            # For AA sequences (embedding or input to translation)
            s = re.sub(r"[UZOB]", "X", str(seq).upper())
            s_spaced = " ".join(list(s))
            processed_sequences.append(f"<AA2fold> {s_spaced}")
        elif sequence_type == '3di':
            # For 3Di sequences (embedding)
            s = str(seq).lower()
            s_spaced = " ".join(list(s))
            processed_sequences.append(f"<fold2AA> {s_spaced}")
        else:
            raise ValueError(f"Unknown sequence_type: {sequence_type}")
    return processed_sequences


def generate_prostt5_embeddings(encoder_model, tokenizer, original_sequences, sequence_type, device, batch_size):
    """
    Generate ProstT5 embeddings for a list of sequences (AA or 3Di).
    Returns embeddings sorted by original sequence length and the corresponding sorted indices.
    """
    all_embeddings = []
    
    # Store original sequences with their original indices and lengths
    # Filter out any non-string or empty sequences that might cause errors
    indexed_sequences = []
    for i, seq_str in enumerate(original_sequences):
        if isinstance(seq_str, str) and len(seq_str) > 0:
            indexed_sequences.append({'seq': seq_str, 'id': i, 'len': len(seq_str)})
        else:
            logging.warning(f"Skipping invalid sequence at index {i}: {seq_str}")

    if not indexed_sequences:
        return np.array([]), []

    # Sort sequences by length for more efficient batching (longest first)
    indexed_sequences.sort(key=lambda x: x['len'], reverse=True)
    
    sorted_original_sequences = [item['seq'] for item in indexed_sequences]
    sorted_indices = [item['id'] for item in indexed_sequences] # Original indices, in sorted order

    with torch.inference_mode():
        for i in tqdm(range(0, len(sorted_original_sequences), batch_size), desc=f"Embedding {sequence_type}"):
            batch_original_seqs = sorted_original_sequences[i:i + batch_size]
            
            batch_preprocessed_seqs = preprocess_sequences_for_prostt5(batch_original_seqs, sequence_type)
            
            ids = tokenizer.batch_encode_plus(
                batch_preprocessed_seqs, add_special_tokens=True, padding="longest", return_tensors="pt"
            )
            input_ids = ids['input_ids'].to(device)
            attention_mask = ids['attention_mask'].to(device)
            
            embedding_repr = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            
            for j, original_seq_in_batch in enumerate(batch_original_seqs):
                original_len = len(original_seq_in_batch)
                # ProstT5 embedding extraction: skip prefix token(s), take up to original_len
                # The first token (index 0) is often a start/control token from the prefix or BOS.
                # We extract `original_len` embeddings, starting from index 1 up to `original_len + 1`.
                seq_emb = embedding_repr.last_hidden_state[j, 1:original_len + 1]
                if seq_emb.shape[0] == 0: # Handle case where sequence might be too short or problematic
                    logging.warning(f"Sequence '{original_seq_in_batch}' resulted in empty embedding. Using zero vector.")
                    per_protein_emb = torch.zeros(encoder_model.config.hidden_size, device=device)
                else:
                    per_protein_emb = seq_emb.mean(dim=0)
                all_embeddings.append(per_protein_emb.cpu().numpy())
                
    return np.array(all_embeddings), sorted_indices


def translate_aa_to_3di(translation_model, tokenizer, aa_sequences, device, batch_size):
    """Translate AA sequences to 3Di sequences, returning them in the original input order."""
    translated_3di_sequences_dict = {} # Store by original index

    # Store original sequences with their original indices and lengths
    indexed_aa_sequences = [{'seq': seq, 'id': i, 'len': len(seq)} for i, seq in enumerate(aa_sequences)]
    # Sort by length for efficient batching
    indexed_aa_sequences.sort(key=lambda x: x['len'], reverse=True)

    sorted_input_aa_sequences = [item['seq'] for item in indexed_aa_sequences]
    original_indices_for_sorted = [item['id'] for item in indexed_aa_sequences]

    with torch.inference_mode():
        for i in tqdm(range(0, len(sorted_input_aa_sequences), batch_size), desc="Translating AA to 3Di"):
            batch_aa_seqs_sorted = sorted_input_aa_sequences[i:i + batch_size]
            batch_original_indices = original_indices_for_sorted[i:i + batch_size]

            # Preprocess for translation (input to ProstT5 is <AA2fold> ...)
            batch_preprocessed_for_translation = preprocess_sequences_for_prostt5(batch_aa_seqs_sorted, 'aa')
            
            ids = tokenizer.batch_encode_plus(
                batch_preprocessed_for_translation, add_special_tokens=True, padding="longest", return_tensors="pt"
            )
            input_ids = ids['input_ids'].to(device)
            attention_mask = ids['attention_mask'].to(device)

            # Set max_length for generation. Should be close to original AA length.
            # Add a small buffer. Using model's default if not set.
            # max_gen_length = input_ids.shape[1] + 20 # Heuristic based on tokenized input length
            
            generated_ids = translation_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max(len(s) for s in batch_aa_seqs_sorted) + 30, # Max length of original AA in batch + buffer
                do_sample=False # Use beam search for more deterministic output
            )
            
            decoded_raw_3di = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for k, raw_3di in enumerate(decoded_raw_3di):
                cleaned_3di = raw_3di.replace(" ", "").lower()
                original_idx = batch_original_indices[k]
                translated_3di_sequences_dict[original_idx] = cleaned_3di
    
    # Reconstruct in original order
    final_translated_sequences = [translated_3di_sequences_dict[j] for j in range(len(aa_sequences))]
    return final_translated_sequences


def process_split_data_prostt5(df_split, split_name, output_dir, 
                               encoder_model, translation_model, tokenizer, 
                               device, batch_size):
    """Process data for a specific split: generate AA and 3Di-from-AA embeddings."""
    logging.info(f"\nProcessing {split_name} split with {len(df_split)} sequences for ProstT5...")
    
    aa_sequences_original_order = df_split['sequence'].tolist()
    sequence_ids_original_order = df_split['sequence_id'].tolist()
    sf_labels_original_order = df_split['SF'].values

    # 1. Generate AA Embeddings
    logging.info(f"Generating AA embeddings for {split_name}...")
    # aa_embeddings_sorted_by_length will have embeddings for sequences sorted by length.
    # aa_sort_indices contains the original indices of these sequences.
    aa_embeddings_sorted_by_length, aa_sort_indices = generate_prostt5_embeddings(
        encoder_model, tokenizer, aa_sequences_original_order, 'aa', device, batch_size
    )
    
    if aa_embeddings_sorted_by_length.size == 0:
        logging.warning(f"No AA embeddings generated for split {split_name}. Skipping saving AA embeddings.")
    else:
        # Metadata should correspond to the order of aa_embeddings_sorted_by_length
        aa_metadata_ids_sorted = [sequence_ids_original_order[i] for i in aa_sort_indices]
        aa_metadata_sf_sorted = sf_labels_original_order[aa_sort_indices]

        output_aa_emb_file = output_dir / f'prostT5_aa_embeddings_{split_name}.npz'
        np.savez_compressed(output_aa_emb_file, embeddings=aa_embeddings_sorted_by_length)
        
        aa_metadata_df = pd.DataFrame({
            'sequence_id': aa_metadata_ids_sorted,
            'SF': aa_metadata_sf_sorted
        })
        output_aa_labels_file = output_dir / f'prostT5_aa_labels_{split_name}.csv'
        aa_metadata_df.to_csv(output_aa_labels_file, index=False)
        logging.info(f"Saved AA embeddings to {output_aa_emb_file} (shape: {aa_embeddings_sorted_by_length.shape})")
        logging.info(f"Saved AA metadata to {output_aa_labels_file}")

    # 2. Translate AA to 3Di
    logging.info(f"Translating AA to 3Di for {split_name}...")
    # translated_3di_sequences are in the *same order* as aa_sequences_original_order
    translated_3di_sequences = translate_aa_to_3di(
        translation_model, tokenizer, aa_sequences_original_order, device, batch_size
    )

    # 3. Generate 3Di Embeddings (from translated 3Di)
    logging.info(f"Generating 3Di embeddings from translated sequences for {split_name}...")
    # three_di_embeddings_sorted_by_length for sequences sorted by length of translated_3di_sequences.
    # three_di_sort_indices refers to indices within translated_3di_sequences (which is in original order).
    three_di_embeddings_sorted_by_length, three_di_sort_indices = generate_prostt5_embeddings(
        encoder_model, tokenizer, translated_3di_sequences, '3di', device, batch_size
    )

    if three_di_embeddings_sorted_by_length.size == 0:
        logging.warning(f"No 3Di embeddings generated for split {split_name}. Skipping saving 3Di embeddings.")
    else:
        # Metadata for 3Di embeddings should correspond to the original AA sequences.
        # three_di_sort_indices are indices into translated_3di_sequences.
        # Since translated_3di_sequences is aligned with aa_sequences_original_order,
        # these indices can be used on sequence_ids_original_order and sf_labels_original_order.
        three_di_metadata_ids_sorted = [sequence_ids_original_order[i] for i in three_di_sort_indices]
        three_di_metadata_sf_sorted = sf_labels_original_order[three_di_sort_indices]

        output_3di_emb_file = output_dir / f'prostT5_3di_from_aa_embeddings_{split_name}.npz'
        np.savez_compressed(output_3di_emb_file, embeddings=three_di_embeddings_sorted_by_length)

        three_di_metadata_df = pd.DataFrame({
            'sequence_id': three_di_metadata_ids_sorted,
            'SF': three_di_metadata_sf_sorted
        })
        output_3di_labels_file = output_dir / f'prostT5_3di_from_aa_labels_{split_name}.csv'
        three_di_metadata_df.to_csv(output_3di_labels_file, index=False)
        logging.info(f"Saved 3Di (from AA) embeddings to {output_3di_emb_file} (shape: {three_di_embeddings_sorted_by_length.shape})")
        logging.info(f"Saved 3Di (from AA) metadata to {output_3di_labels_file}")
    
    if aa_embeddings_sorted_by_length.size > 0: # Check if aa_metadata_sf_sorted exists
         logging.info(f"Number of unique superfamilies in {split_name} split: {len(np.unique(aa_metadata_sf_sorted))}")


def main():
    parser = argparse.ArgumentParser(description='Generate ProstT5 AA and 3Di-from-AA embeddings for protein sequences')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/s30_full.parquet',
                        help='Input parquet file containing protein sequences (default: data/TED/s30/s30_full.parquet)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/prostT5_embeddings',
                        help='Output directory for embeddings (default: data/TED/s30/prostT5_embeddings)')
    parser.add_argument('--batch-size', '-b', type=int, default=32, # Smaller default due to larger models
                        help='Batch size for embedding generation (default: 32)')
    parser.add_argument('--splits', '-s', nargs='+', choices=['train', 'val', 'test'], 
                        help='Process specific splits (e.g., train val test). If not specified, processes all splits found in the input file.')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    encoder_model, translation_model, tokenizer, device = get_prostT5_models_and_tokenizer()
    
    logging.info(f"Loading data from {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        logging.error(f"Failed to load input Parquet file: {args.input}. Error: {e}")
        return

    if 'sequence' not in df.columns or 'sequence_id' not in df.columns or \
       'SF' not in df.columns or 'split' not in df.columns:
        logging.error("Input Parquet file must contain 'sequence', 'sequence_id', 'SF', and 'split' columns.")
        return
        
    splits_to_process = args.splits if args.splits else df['split'].unique()
    
    for split in splits_to_process:
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) > 0:
            process_split_data_prostt5(df_split, split, output_dir, 
                                       encoder_model, translation_model, tokenizer, 
                                       device, args.batch_size)
        else:
            logging.warning(f"No data found for split '{split}'. Skipping.")

if __name__ == "__main__":
    main()