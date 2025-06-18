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
- Loads models sequentially to manage GPU memory.

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


def get_prostT5_tokenizer_and_device():
    """Load ProstT5 tokenizer and determine device."""
    logging.info("Loading ProstT5 tokenizer and determining device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model_name = "Rostlab/ProstT5"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    logging.info("ProstT5 tokenizer and device configured.")
    return tokenizer, device


def load_prostT5_model(device):
    """Load ProstT5 Seq2Seq model."""
    logging.info("Loading ProstT5 Seq2Seq model...")
    model_name = "Rostlab/ProstT5"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    if device.type == 'cuda':
        logging.info("Using half precision for model on GPU.")
        model.half()
    else:
        logging.info("Using full precision for model on CPU.")
        model.float()
    
    logging.info("ProstT5 model loaded.")
    return model


def _sort_sequences_by_length(sequences):
    """
    Sorts sequences by length in descending order.
    
    Returns a tuple of:
    - A list of sorted sequences.
    - A list of the original indices corresponding to the sorted sequences.
    """
    indexed_sequences = []
    for i, seq_str in enumerate(sequences):
        if isinstance(seq_str, str) and len(seq_str) > 0:
            indexed_sequences.append({'seq': seq_str, 'id': i, 'len': len(seq_str)})
        else:
            logging.warning(f"Skipping invalid sequence at index {i}: {seq_str}")

    if not indexed_sequences:
        return [], []

    indexed_sequences.sort(key=lambda x: x['len'], reverse=True)
    
    sorted_sequences = [item['seq'] for item in indexed_sequences]
    sorted_indices = [item['id'] for item in indexed_sequences]
    return sorted_sequences, sorted_indices


def _save_results(output_dir, embed_type, split_name, embeddings, metadata_ids, metadata_sf):
    """Saves embeddings and corresponding metadata to disk."""
    if embeddings.size == 0:
        logging.warning(f"No {embed_type} embeddings were generated for split {split_name}. Skipping save.")
        return

    file_prefix = f"prostT5_{'3di_from_aa' if embed_type == '3di' else 'aa'}"
    
    output_emb_file = output_dir / f'{file_prefix}_embeddings_{split_name}.npz'
    np.savez_compressed(output_emb_file, embeddings=embeddings)
    
    metadata_df = pd.DataFrame({'sequence_id': metadata_ids, 'SF': metadata_sf})
    output_labels_file = output_dir / f'{file_prefix}_labels_{split_name}.csv'
    metadata_df.to_csv(output_labels_file, index=False)
    
    logging.info(f"Saved {embed_type} embeddings to {output_emb_file} (shape: {embeddings.shape})")
    logging.info(f"Saved {embed_type} metadata to {output_labels_file}")


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
    
    sorted_original_sequences, sorted_indices = _sort_sequences_by_length(original_sequences)

    if not sorted_original_sequences:
        return np.array([]), []

    # Pre-process all sorted sequences at once before batching
    all_preprocessed_sorted_sequences = preprocess_sequences_for_prostt5(sorted_original_sequences, sequence_type)

    with torch.inference_mode():
        for i in tqdm(range(0, len(all_preprocessed_sorted_sequences), batch_size), desc=f"Embedding {sequence_type}"):
            # batch_original_seqs is needed for original_len, so we still need to slice it.
            batch_original_seqs = sorted_original_sequences[i:i + batch_size]
            batch_preprocessed_seqs = all_preprocessed_sorted_sequences[i:i + batch_size]
            
            try:
                ids = tokenizer.batch_encode_plus(
                    batch_preprocessed_seqs, add_special_tokens=True, padding="longest", return_tensors="pt"
                )
                input_ids = ids['input_ids'].to(device)
                attention_mask = ids['attention_mask'].to(device)
                
                embedding_repr = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
                
                for j, original_seq_in_batch in enumerate(batch_original_seqs):
                    original_len = len(original_seq_in_batch)
                    seq_emb = embedding_repr.last_hidden_state[j, 1:original_len + 1]
                    if seq_emb.shape[0] == 0:
                        logging.warning(f"Sequence '{original_seq_in_batch}' resulted in empty embedding. Using zero vector.")
                        per_protein_emb = torch.zeros(encoder_model.config.hidden_size, device=device)
                    else:
                        per_protein_emb = seq_emb.mean(dim=0)
                    all_embeddings.append(per_protein_emb.cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.error(f"CUDA OOM Error during batch processing (Batch start index: {i}, type: {sequence_type}). "
                                  f"Try reducing batch size further. Batch sequences (first 5 lengths): "
                                  f"{[len(s) for s in batch_original_seqs[:5]]}")
                    # Skip this batch or re-raise to stop execution. For now, re-raise.
                    raise e 
                else:
                    raise e # Re-raise other runtime errors
                
    return np.array(all_embeddings), sorted_indices


def translate_aa_to_3di(translation_model, tokenizer, aa_sequences, device, batch_size):
    """Translate AA sequences to 3Di sequences, returning them in the original input order."""
    translated_3di_sequences_dict = {} 

    sorted_input_aa_sequences, original_indices_for_sorted = _sort_sequences_by_length(aa_sequences)
    
    if not sorted_input_aa_sequences:
        logging.warning("No valid AA sequences provided for translation.")
        return ["" for _ in aa_sequences]

    # Pre-process all sorted AA sequences at once before batching for translation
    all_preprocessed_sorted_aa_sequences = preprocess_sequences_for_prostt5(sorted_input_aa_sequences, 'aa')

    with torch.inference_mode():
        for i in tqdm(range(0, len(all_preprocessed_sorted_aa_sequences), batch_size), desc="Translating AA to 3Di"):
            # batch_aa_seqs_sorted is needed for max_length calculation, so slice original sorted.
            batch_aa_seqs_sorted = sorted_input_aa_sequences[i:i + batch_size]
            batch_original_indices = original_indices_for_sorted[i:i + batch_size]
            batch_preprocessed_for_translation = all_preprocessed_sorted_aa_sequences[i:i + batch_size]
            
            try:
                ids = tokenizer.batch_encode_plus(
                    batch_preprocessed_for_translation, add_special_tokens=True, padding="longest", return_tensors="pt"
                )
                input_ids = ids['input_ids'].to(device)
                attention_mask = ids['attention_mask'].to(device)
                
                generated_ids = translation_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max(len(s) for s in batch_aa_seqs_sorted) + 50, # Increased buffer slightly
                    do_sample=False 
                )
                
                decoded_raw_3di = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                for k, raw_3di in enumerate(decoded_raw_3di):
                    cleaned_3di = raw_3di.replace(" ", "").lower()
                    original_idx = batch_original_indices[k]
                    translated_3di_sequences_dict[original_idx] = cleaned_3di
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.error(f"CUDA OOM Error during translation (Batch start index: {i}). "
                                  f"Try reducing batch size further. Batch sequences (first 5 lengths): "
                                  f"{[len(s) for s in batch_aa_seqs_sorted[:5]]}")
                    # Populate with empty strings for this batch to maintain structure
                    for k_idx in batch_original_indices:
                        translated_3di_sequences_dict[k_idx] = "" # Placeholder for failed translations
                else:
                    raise e

    final_translated_sequences = [translated_3di_sequences_dict.get(j, "") for j in range(len(aa_sequences))]
    return final_translated_sequences


def process_split_data_prostt5(df_split, split_name, output_dir, 
                               tokenizer, device, batch_size):
    logging.info(f"\nProcessing {split_name} split with {len(df_split)} sequences for ProstT5...")
    
    aa_sequences_original_order = df_split['sequence'].tolist()
    sequence_ids_original_order = df_split['sequence_id'].tolist()
    sf_labels_original_order = df_split['SF'].values

    # Load the main ProstT5 model once. We can get the encoder from it.
    model = load_prostT5_model(device)
    encoder_model = model.get_encoder()

    # 1. Generate AA Embeddings
    logging.info(f"Generating AA embeddings for {split_name}...")
    aa_embeddings_sorted, aa_sort_indices = generate_prostt5_embeddings(
        encoder_model, tokenizer, aa_sequences_original_order, 'aa', device, batch_size
    )
    
    if aa_embeddings_sorted.size > 0:
        aa_metadata_ids_sorted = [sequence_ids_original_order[i] for i in aa_sort_indices]
        aa_metadata_sf_sorted = sf_labels_original_order[aa_sort_indices]
        _save_results(output_dir, 'aa', split_name, aa_embeddings_sorted, 
                      aa_metadata_ids_sorted, aa_metadata_sf_sorted)
        logging.info(f"Number of unique superfamilies in {split_name} split (based on AA): {len(np.unique(aa_metadata_sf_sorted))}")

    # 2. Translate AA to 3Di
    logging.info(f"Translating AA to 3Di for {split_name}...")
    translated_3di_sequences = translate_aa_to_3di(
        model, tokenizer, aa_sequences_original_order, device, batch_size
    )
    
    # 3. Generate 3Di Embeddings (from translated 3Di)
    # Filter out sequences that might have failed translation (are empty strings)
    valid_translations = [
        (seq, original_idx) for original_idx, seq in enumerate(translated_3di_sequences) if seq and isinstance(seq, str) and len(seq) > 0
    ]

    if not valid_translations:
        logging.warning(f"No valid 3Di sequences to embed for {split_name} after translation. Skipping 3Di embeddings.")
    else:
        valid_3di_sequences = [item[0] for item in valid_translations]
        original_indices_of_valid = [item[1] for item in valid_translations]

        logging.info(f"Generating 3Di embeddings for {len(valid_3di_sequences)} translated sequences for {split_name}...")
        
        # Note: three_di_sort_indices_relative are indices relative to the `valid_3di_sequences` list
        three_di_embeddings_sorted, three_di_sort_indices_relative = generate_prostt5_embeddings(
            encoder_model, tokenizer, valid_3di_sequences, '3di', device, batch_size
        )

        if three_di_embeddings_sorted.size > 0:
            # Map relative sort indices back to original full list indices
            original_indices_sorted = [original_indices_of_valid[i] for i in three_di_sort_indices_relative]
            
            three_di_metadata_ids_sorted = [sequence_ids_original_order[i] for i in original_indices_sorted]
            three_di_metadata_sf_sorted = sf_labels_original_order[original_indices_sorted]

            _save_results(output_dir, '3di', split_name, three_di_embeddings_sorted, 
                          three_di_metadata_ids_sorted, three_di_metadata_sf_sorted)

    # Clean up the model and clear GPU cache
    del model
    del encoder_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Generate ProstT5 AA and 3Di-from-AA embeddings for protein sequences')
    parser.add_argument('--input', '-i', type=str, default='data/TED/s30/s30_full.parquet',
                        help='Input parquet file containing protein sequences (default: data/TED/s30/s30_full.parquet)')
    parser.add_argument('--output', '-o', type=str, default='data/TED/s30/prostT5_embeddings',
                        help='Output directory for embeddings (default: data/TED/s30/prostT5_embeddings)')
    parser.add_argument('--batch-size', '-b', type=int, default=32, 
                        help='Batch size for embedding generation (default: 32). Reduce if OOM errors occur.')
    parser.add_argument('--splits', '-s', nargs='+', choices=['train', 'val', 'test'], 
                        help='Process specific splits. If not specified, processes all splits found.')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer, device = get_prostT5_tokenizer_and_device()
    
    logging.info(f"Loading data from {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        logging.error(f"Failed to load input Parquet file: {args.input}. Error: {e}")
        return

    required_cols = ['sequence', 'sequence_id', 'SF', 'split']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Input Parquet file must contain {required_cols} columns.")
        return
        
    splits_to_process = args.splits if args.splits else df['split'].unique()
    
    for split in splits_to_process:
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) > 0:
            try:
                process_split_data_prostt5(df_split, split, output_dir, 
                                           tokenizer, device, args.batch_size)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.error(f"An OOM error occurred while processing split '{split}'. "
                                  "Consider reducing --batch-size or checking for extremely long sequences.")
                    # Optionally, continue to the next split or re-raise
                    # For now, we'll log and continue to allow other splits to process if possible
                    logging.warning(f"Skipping rest of split '{split}' due to OOM error.")
                    continue 
                else:
                    logging.error(f"A runtime error occurred processing split '{split}': {e}")
                    raise # Re-raise other runtime errors
        else:
            logging.warning(f"No data found for split '{split}'. Skipping.")

if __name__ == "__main__":
    main()