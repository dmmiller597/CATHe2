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
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from transformers import T5EncoderModel, T5TokenizerFast
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

def _preprocess(seq: str) -> str:
    """Replace rare AAs with X and insert spaces for the tokenizer."""
    return " ".join(_aa_cleaner.sub("X", seq))

# ───────────────────────── model utilities ─────────────────────────

def _load_prott5() -> Tuple[T5EncoderModel, T5TokenizerFast, torch.device]:
    name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    log.info("🔌 Loading ProtT5-XL-U50 encoder (%s)", name)
    mdl = T5EncoderModel.from_pretrained(name)
    tok = T5TokenizerFast.from_pretrained(name, do_lower_case=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(device).eval()  # fp16 weights already baked in

    if hasattr(torch, "compile"):
        log.info("🚀  Compiling model with torch.compile() for speed-up")
        mdl = torch.compile(mdl)
    return mdl, tok, device

@torch.inference_mode()
def _encode_batch(
    model: T5EncoderModel,
    tok: T5TokenizerFast,
    device: torch.device,
    seqs: List[str],
) -> np.ndarray:
    """Mean-pool ProtT5 encoder features → (B, 1024) float16."""
    if not seqs:
        return np.empty((0, 1024), dtype=np.float16)

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    rev = np.argsort(order)

    tok_inputs = tok(
        [_preprocess(seqs[i]) for i in order],
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

def _iter_token_batches(
    seqs: List[str],
    labels: np.ndarray,
    budget: int,
):
    """Yield subsequences whose combined (len + 2) ≤ token budget."""
    buf_s: List[str] = []
    buf_l: List[int] = []
    running = 0

    for s, lb in zip(seqs, labels):
        t = len(s) + 2
        if buf_s and running + t > budget:
            yield buf_s, np.asarray(buf_l, dtype=np.int32)
            buf_s, buf_l, running = [], [], 0
        buf_s.append(s)
        buf_l.append(lb)
        running += t

    if buf_s:
        yield buf_s, np.asarray(buf_l, dtype=np.int32)

# ───────────────────────────── I/O helpers ─────────────────────────

def _create_memmap(path: Path, shape: Tuple[int, ...], dtype: str) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("📝  Allocating memmap %s  shape=%s  dtype=%s", path, shape, dtype)
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)

# ─────────────────────────── argument parsing ──────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="data/TED/s90/s90_meta.parquet", help="Parquet with sequence_id + label column")
    p.add_argument("--seq",  default="data/TED/s90_reps.parquet", help="Parquet/Arrow with sequence_id + sequence")
    p.add_argument("--out",  default="data/TED/s90/embeddings/protT5", help="Output directory for .npy files")

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

    # 1) Load metadata once and index for O(1) look-ups
    meta = pd.read_parquet(args.meta).set_index("sequence_id")
    if args.label_col not in meta.columns:
        log.error("❌  --meta missing required label column '%s'", args.label_col)
        sys.exit(1)

    labels_arr = meta[args.label_col].to_numpy("int32", copy=False)
    n_total = len(meta)
    log.info("📊  Total sequences according to metadata: %s", f"{n_total:,}")

    # 2) Prepare output mem-maps
    emb_mm = _create_memmap(out_dir / "embeddings.npy", (n_total, 1024), "float16")
    lab_mm = _create_memmap(out_dir / "labels.npy",     (n_total,),       "int32")

    # 3) Load model
    model, tokenizer, device = _load_prott5()

    # 4) Stream sequence dataset
    ds_seq = ds.dataset(args.seq, format="parquet")
    scanner = ds_seq.scanner(columns=["sequence_id", "sequence"],
                             batch_size=args.arrow_batch)

    written = 0
    for record_batch in tqdm(scanner.to_batches(), unit_scale=args.arrow_batch, desc="Embedding"):
        ids = record_batch.column("sequence_id").to_pylist()
        seqs = record_batch.column("sequence").to_pylist()

        meta_slice = meta.reindex(ids)
        mask = meta_slice[args.label_col].notna()
        if not mask.all():
            seqs   = [s for s, ok in zip(seqs, mask) if ok]
            labels = meta_slice.loc[mask, args.label_col].to_numpy("int32", copy=False)
        else:
            labels = meta_slice[args.label_col].to_numpy("int32", copy=False)

        for sub_seqs, sub_labels in _iter_token_batches(seqs, labels, args.batch_tokens):
            embs = _encode_batch(model, tokenizer, device, sub_seqs)
            n = len(embs)
            emb_mm[written:written + n] = embs
            lab_mm[written:written + n] = sub_labels
            written += n

        if written and written % args.flush_every == 0:
            emb_mm.flush(); lab_mm.flush();
            log.info("💾  Flushed to disk @ %s sequences", f"{written:,}")

    # 5) Final flush & sanity check
    emb_mm.flush(); lab_mm.flush()
    assert written == n_total, f"Wrote {written} vs expected {n_total}"
    log.info("✅  Finished embedding %s sequences → %s", f"{written:,}", out_dir)


if __name__ == "__main__":
    main()