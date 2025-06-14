#!/usr/bin/env python3
# scripts/embeddings_s90_prott5.py
"""
ProtT5 embedding pipeline
======================================================================


Inputs
------
  --meta : Parquet file with columns  [sequence_id, label]
  --seq  : Parquet/Arrow dataset with [sequence_id, sequence]
  --out  : Output directory (will contain embeddings.npy, labels.npy)

Outputs
-------
  <out>/embeddings.npy  (N, 1024) float16  – mean-pooled ProtT5 features
  <out>/labels.npy      (N,)      int32    – numeric labels from --meta

Important flags
--------------
  --batch-tokens  Max tokens (<len> + 2) per GPU micro-batch  (default: 120k)
  --arrow-batch   Rows pulled per Arrow scan batch            (default: 64k)
  --flush-every   Flush mem-maps every N written sequences    (default: 1M)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

# ───────────────────────────── logger ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("protT5_embed")

# ──────────────────── amino-acid preprocessing ─────────────────────
_aa_cleaner = re.compile(r"[UZOB]")  # ambiguous/rare residues → X

def _preprocess(seq: str, cleaner: re.Pattern) -> str:
    """Replace rare AAs with X and insert spaces for the tokenizer."""
    return " ".join(cleaner.sub("X", seq))

# ───────────────────────── model utilities ─────────────────────────

def _load_prott5() -> Tuple[T5EncoderModel, T5Tokenizer, torch.device]:
    name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    log.info("🔌 Loading ProtT5-XL-U50 encoder (%s)", name)
    mdl = T5EncoderModel.from_pretrained(name)
    tok = T5Tokenizer.from_pretrained(name, do_lower_case=False, legacy=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(device).eval()  # fp16 weights already baked in

    if hasattr(torch, "compile"):
        log.info("🚀  Compiling model with torch.compile() for speed-up")
        mdl = torch.compile(mdl)
    return mdl, tok, device

@torch.inference_mode()
def _encode_batch(
    model: T5EncoderModel,
    tok: T5Tokenizer,
    device: torch.device,
    seqs: List[str],
    cleaner: re.Pattern,
) -> np.ndarray:
    """Mean-pool ProtT5 encoder features → (B, 1024) float16."""
    if not seqs:
        return np.empty((0, 1024), dtype=np.float16)

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    rev = np.argsort(order)

    tok_inputs = tok(
        [_preprocess(seqs[i], cleaner) for i in order],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )

    ids = tok_inputs.input_ids.to(device)
    att = tok_inputs.attention_mask.to(device)

    h = model(input_ids=ids, attention_mask=att).last_hidden_state  # (B, L, 1024)

    lengths = att.sum(1)                    # includes <s> and </s>
    mask = att.bool()
    mask[:, 0] = False                      # drop <s>
    mask.scatter_(1, (lengths - 1).unsqueeze(1), False)  # drop </s>

    pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(1)
    return pooled.to(torch.float16).cpu().numpy()[rev]

# ─────────────────── variable-token batching helper ─────────────────

def _iter_batches_from_arrow(
    scanner: ds.Scanner,
    meta_map: dict[str, int],
    token_budget: int,
    label_col: str,
) -> Iterator[Tuple[int, List[str], np.ndarray]]:
    """
    Generator that streams from an Arrow scanner, groups sequences into
    token-limited batches, and yields batch data for processing.

    Yields:
        A tuple of (start_index, sequences, labels) for each batch.
    """
    buf_seqs: List[str] = []
    buf_labels: List[int] = []
    buf_indices: List[int] = []
    running_tokens = 0
    start_offset = 0

    for i, record in enumerate(scanner.to_reader()):
        chunk = record.read_next_batch()
        ids = chunk.column("sequence_id").to_pylist()
        seqs = chunk.column("sequence").to_pylist()

        for j, (seq_id, seq) in enumerate(zip(ids, seqs)):
            label = meta_map.get(seq_id)
            if label is None:
                continue

            n_tokens = len(seq) + 2
            if buf_seqs and running_tokens + n_tokens > token_budget:
                yield start_offset, buf_seqs, np.array(buf_labels, dtype=np.int32)
                start_offset += len(buf_seqs)
                buf_seqs, buf_labels, running_tokens = [], [], 0

            buf_seqs.append(seq)
            buf_labels.append(label)
            running_tokens += n_tokens
    
    if buf_seqs:
        yield start_offset, buf_seqs, np.array(buf_labels, dtype=np.int32)

# ─────────────────────────── argument parsing ──────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", default="data/TED/s90/s90_meta.parquet", help="Parquet with sequence_id + label column")
    p.add_argument("--seq",  default="data/TED/s90_reps.parquet", help="Parquet/Arrow with sequence_id + sequence")
    p.add_argument("--out", "--output-dir", default="data/TED/s90/embeddings/protT5",
                   help="Output directory for .npy files")

    p.add_argument("--label-col", default="SF_label", help="Column in --meta to use as label")
    p.add_argument("--batch-tokens", type=int, default=120_000,
                   help="Max tokens per GPU micro-batch (len+2)")
    p.add_argument("--arrow-batch",  type=int, default=64_000,
                   help="Rows fetched per Arrow scan batch")
    p.add_argument("--flush-every", type=int, default=1_000_000,
                   help="Flush memmaps every N sequences")
    return p.parse_args()

# ─────────────────────────────── main ──────────────────────────────

def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaner = re.compile(r"[UZOB]")

    # 1) Load metadata into a memory-efficient dictionary for fast lookups
    log.info("📖  Loading metadata from %s", args.meta)
    meta_table = ds.dataset(args.meta).to_table(columns=["sequence_id", args.label_col])
    meta_map = dict(zip(
        meta_table["sequence_id"].to_pylist(),
        meta_table[args.label_col].to_pylist()
    ))
    n_total = len(meta_map)
    log.info("📊  Total sequences to find: %s", f"{n_total:,}")

    # 2) Prepare output mem-maps
    emb_path = out_dir / "embeddings.npy"
    lab_path = out_dir / "labels.npy"
    log.info("📝  Allocating memmap %s", emb_path)
    emb_mm = np.memmap(emb_path, mode="w+", dtype="float16", shape=(n_total, 1024))
    log.info("📝  Allocating memmap %s", lab_path)
    lab_mm = np.memmap(lab_path, mode="w+", dtype="int32", shape=(n_total,))

    # 3) Load model
    model, tokenizer, device = _load_prott5()

    # 4) Stream sequence dataset and process in batches
    seq_ds = ds.dataset(args.seq, format="parquet")
    scanner = seq_ds.scanner(
        columns=["sequence_id", "sequence"],
        batch_size=args.arrow_batch
    )
    
    total_sequences = seq_ds.count_rows()
    pbar = tqdm(
        _iter_batches_from_arrow(scanner, meta_map, args.batch_tokens, args.label_col),
        desc="Embedding",
        unit=" seq",
        total=total_sequences
    )

    written_count = 0
    for start_idx, seqs, labels in pbar:
        embs = _encode_batch(model, tokenizer, device, seqs, cleaner)
        n_current = len(embs)
        end_idx = start_idx + n_current
        emb_mm[start_idx:end_idx] = embs
        lab_mm[start_idx:end_idx] = labels
        written_count = end_idx
        pbar.update(n_current)
        pbar.set_postfix(sequences=f"{written_count:,}/{n_total:,}")


    # 5) Final flush & sanity check
    emb_mm.flush()
    lab_mm.flush()
    assert written_count == n_total, f"Wrote {written_count} vs expected {n_total}"
    log.info("✅  Finished embedding %s sequences → %s", f"{written_count:,}", out_dir)


if __name__ == "__main__":
    main()