#!/usr/bin/env python3
"""Efficient ProtT5 mean-pool embeddings written directly to a memory-mapped array.

Usage example:
    python -m cathe2.scripts.embed_prott5 \
        --fasta data/TED/s30/s30_full.fasta \
        --meta  data/TED/s30/s30_full.json \
        --out-dir data/TED/s30/protT5 \
        --device cuda:0 \
        --max-tokens 100000 

The script produces three artefacts in *out-dir*:
    ├─ protT5_embeddings.npy          # N × 1024 float mem-map
    ├─ protT5_id_index_map.json       # sequence-id → row index
    └─ ../splits/{train,val,test}.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
import itertools

_LOGGER = logging.getLogger(__name__)

# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # will be determined by args

    # model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, legacy=True)

    return model, tokenizer


def read_fasta_ids(path: Path) -> List[str]:
    """Parse a (multi-)FASTA file and return only the IDs."""
    ids: List[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def read_fasta_generator(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (id, sequence) tuples from a FASTA file."""
    seq_buf: List[str] = []
    curr_id = None

    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr_id is not None:
                    yield curr_id, "".join(seq_buf)
                    seq_buf.clear()
                curr_id = line[1:].split()[0]
            else:
                seq_buf.append(line)
        # final record
        if curr_id is not None and seq_buf:
            yield curr_id, "".join(seq_buf)


def load_meta_from_dir(path: Path) -> dict:
    """Load all JSON files in a directory and merge them into a meta dict.

    The split name is derived from the file name (e.g., 'test.json' -> 'test').
    """
    meta = {}
    if not path.is_dir():
        _LOGGER.warning("--meta-dir must be a directory. No metadata loaded.")
        return meta

    for json_path in path.glob("*.json"):
        split = json_path.stem
        with json_path.open() as fh:
            split_data = json.load(fh)
            for key, value in split_data.items():
                if key in meta:
                    _LOGGER.warning("Duplicate ID %s found in split %s, overwriting.", key, split)
                meta[key] = {"split": split, "sf": value}
    return meta


# ────────────────────────────────────────────────────────────────────────────────
# Dynamic batching utilities
# ────────────────────────────────────────────────────────────────────────────────


def _iter_token_batches(
    seqs: List[str],
    idxs: List[int],
    token_budget: int,
) -> Iterator[Tuple[List[int], List[str]]]:
    """Yield *(row_indices, sequences)* whose combined tokens ≤ *token_budget*.

    Token count is approximated as len(seq) + 2 (special tokens).
    """

    buffer_ids: List[int] = []
    buffer_seqs: List[str] = []
    running = 0

    for i, seq in zip(idxs, seqs):
        tokens = len(seq) + 2  # <s> and </s>
        if buffer_seqs and running + tokens > token_budget:
            yield buffer_ids, buffer_seqs
            buffer_ids, buffer_seqs, running = [], [], 0
        buffer_ids.append(i)
        buffer_seqs.append(seq)
        running += tokens

    if buffer_seqs:
        yield buffer_ids, buffer_seqs

# ────────────────────────────────────────────────────────────────────────────────
# Embedding core
# ────────────────────────────────────────────────────────────────────────────────


def embed_sequences_stream(
    fasta_path: Path,
    id_to_idx: dict[str, int],
    memmap: np.memmap,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    max_tokens: int,
    chunk_size: int = 100_000,
):
    """Encode and write each protein's mean embedding into *memmap* in-place."""

    _LOGGER.info("Embedding %d sequences", len(id_to_idx))
    pbar = tqdm(total=len(id_to_idx), unit="seq")

    seq_generator = read_fasta_generator(fasta_path)

    start_time = time.time()
    total_sequences_processed = 0

    while True:
        chunk = list(itertools.islice(seq_generator, chunk_size))
        if not chunk:
            break

        ids_chunk, seqs_chunk = zip(*chunk)

        # 1️⃣ order sequences by length (descending) for better packing
        order = sorted(range(len(seqs_chunk)), key=lambda i: len(seqs_chunk[i]), reverse=True)

        seqs_sorted = [seqs_chunk[i] for i in order]
        ids_sorted = [ids_chunk[i] for i in order]
        row_indices = [id_to_idx[id_] for id_ in ids_sorted]

        # 2️⃣ clean & whitespace-tokenise
        seqs_tok = [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in seqs_sorted]

        with torch.inference_mode():
            for batch_idx, (row_idxs_for_batch, batch_seqs) in enumerate(
                _iter_token_batches(seqs_tok, row_indices, max_tokens)
            ):
                input_batch = tokenizer.batch_encode_plus(
                    batch_seqs,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt",
                )
                input_batch = {k: v.to(device) for k, v in input_batch.items()}

                reps = model(**input_batch).last_hidden_state  # (B, L, 1024)

                mask = input_batch["attention_mask"]  # (B, L)
                # skip special tokens (<s>, </s>) when pooling
                seq_lens = mask.sum(dim=1)

                for j in range(reps.size(0)):
                    seq_len = seq_lens[j].item()
                    seq_rep = reps[j, 1 : seq_len - 1]  # exclude special tokens
                    mean_vec = seq_rep.mean(dim=0)
                    memmap[row_idxs_for_batch[j]] = mean_vec.cpu().numpy().astype(np.float16)

        processed_in_chunk = len(chunk)
        total_sequences_processed += processed_in_chunk
        pbar.update(processed_in_chunk)

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            seq_per_sec = total_sequences_processed / elapsed_time
            pbar.set_postfix({"seq/s": f"{seq_per_sec:.2f}"})

    pbar.close()


def create_and_save_id_index(fasta_path: Path, out_path: Path) -> dict[str, int]:
    """Scan a FASTA file to create a mapping from sequence ID to index and save it."""
    _LOGGER.info("Scanning FASTA to build ID index from %s. This may take a while for large files...", fasta_path)
    id_to_idx: dict[str, int] = {}
    pbar = tqdm(desc="Scanning FASTA", unit=" seqs")
    with fasta_path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                id_ = line[1:].split()[0]
                if id_ not in id_to_idx:
                    id_to_idx[id_] = len(id_to_idx)
                    pbar.update(1)
    pbar.close()
    _LOGGER.info("Found %d unique sequences.", len(id_to_idx))

    save_json({"id2idx": id_to_idx}, out_path)
    _LOGGER.info("ID index map saved to %s", out_path)
    return id_to_idx

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────

def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2)


def write_split_files(meta: dict, ids: List[str], out_dir: Path):
    split_to_ids = {}
    for i in ids:
        split = meta[i]["split"] if i in meta else "unspecified"
        split_to_ids.setdefault(split, []).append(i)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, id_list in split_to_ids.items():
        (out_dir / f"{split}.txt").write_text("\n".join(id_list))

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="ProtT5 mem-map embedding generator")
    p.add_argument("--fasta", default="data/TED/cath_ted_gold_sequences_hmmvalidated_qscore_0.7_S100_rep_seq.fasta", type=Path, help="Input FASTA file")
    p.add_argument("--meta-dir", default="data/TED/s100_splits", type=Path, help="Directory with JSON split files (train.json, val.json, test.json)")
    p.add_argument("--out-dir", default="data/TED/s100/protT5", type=Path, help="Directory for outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-tokens", type=int, default=100000, help="Token budget per batch. Adjust based on GPU memory.")
    p.add_argument("--chunk-size", type=int, default=100_000, help="Number of sequences to process from FASTA at a time")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build or load the ID-to-index map
    id_map_path = args.out_dir / "protT5_id_index_map.json"
    if id_map_path.exists():
        _LOGGER.info("Loading existing ID index from %s", id_map_path)
        with id_map_path.open() as fh:
            id_to_idx = json.load(fh)["id2idx"]
    else:
        id_to_idx = create_and_save_id_index(args.fasta, id_map_path)

    ids = list(id_to_idx.keys())
    if len(ids) == 0:
        _LOGGER.error("No sequences found in FASTA – aborting")
        sys.exit(1)

    _LOGGER.info("Loading metadata from %s", args.meta_dir)
    meta = load_meta_from_dir(args.meta_dir)

    # ── allocate mem-map ───────────────────────────────────────────────────────
    mmap_path = args.out_dir / "protT5_embeddings.npy"
    _LOGGER.info("Allocating memory map array at %s with shape (%d, 1024)", mmap_path, len(ids))
    mmap_arr = np.memmap(mmap_path, mode="w+", dtype="float16", shape=(len(ids), 1024))

    _LOGGER.info("Initialising ProtT5 (this can take a minute)…")
    model, tokenizer = get_T5_model()
    model = model.to(device)

    embed_sequences_stream(args.fasta, id_to_idx, mmap_arr, tokenizer, model, device, args.max_tokens, args.chunk_size)

    mmap_arr.flush()  # ensure everything hits disk

    # ── auxiliary artefacts ────────────────────────────────────────────────────
    # The ID map is now saved at the beginning of the script.
    # We only need to write the split files here.
    write_split_files(meta, ids, args.out_dir.parent / "splits")

    _LOGGER.info("Finished. Mem-map stored at %s", mmap_path)


if __name__ == "__main__":
    main() 