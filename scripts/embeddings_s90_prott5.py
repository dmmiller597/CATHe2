#!/usr/bin/env python3
# scripts/embeddings_s90_prott5.py
"""
One-pass, memory-efficient ProtT5 embedding pipeline for the 78 M-row S90 data set.

Outputs (per split)
───────────────────
<out-dir>/<split>/embeddings.npy   # mem-mapped  (N_split, 1024) float16
<out-dir>/<split>/labels.npy       # mem-mapped  (N_split,)     int32

Design highlights
─────────────────
• <2 GB peak RAM   –  loads only the 4-column metadata table once.
• Streaming Arrow scan of the 13 GB sequence parquet – constant memory.
• Single write per sequence – no temporary objects, no post-processing.
• fp16 encoder + torch.inference_mode()  → fits easily on a single A10.

Typical usage
─────────────
python scripts/embeddings_s90_prott5.py \
  --meta   data/TED/s90/s90_meta.parquet \
  --seq    data/TED/s90_reps.parquet \
  --out    data/TED/s90/embeddings/protT5 \
  --batch  64 --arrow-batch 8000
"""

from __future__ import annotations
import argparse, logging, re, sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

# ─────────────────────────────── logger ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("encode_s90")


# ─────────────────────────── model utilities ───────────────────────────
def load_prott5() -> Tuple[T5EncoderModel, T5Tokenizer, torch.device]:
    """FP16 ProtT5 encoder ⇢ device, eval."""
    log.info("🔌 Loading ProtT5-XL-U50 (encoder-only, fp16)")
    mdl = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc"
    )
    tok = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(dev).eval()  # fp16 weights are baked in
    return mdl, tok, dev


aa_cleaner = re.compile(r"[UZOB]")  # ambiguous & rare → X


def preprocess(seq: str) -> str:
    """Replace rare AAs, add whitespace."""
    return " ".join(list(aa_cleaner.sub("X", seq)))


@torch.inference_mode()
def encode_batch(
    model: T5EncoderModel,
    tokenizer: T5Tokenizer,
    device: torch.device,
    seqs: List[str],
) -> np.ndarray:
    """
    Encode ≤64 sequences → (B, 1024) float16 numpy.
    Keeps original order.
    """
    if not seqs:
        return np.empty((0, 1024), dtype=np.float16)

    # length-sort for speed, remember indices to restore order
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    rev_order = np.argsort(order)

    tok_inputs = tokenizer.batch_encode_plus(
        [preprocess(seqs[i]) for i in order],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )

    inp_ids = tok_inputs["input_ids"].to(device)
    attn = tok_inputs["attention_mask"].to(device)

    hs = model(input_ids=inp_ids, attention_mask=attn).last_hidden_state

    # mean-pool (exclude <s> & </s>) per sequence
    lengths = attn.sum(dim=1)  # includes both special tokens
    pooled = [
        hs[j, 1 : ln - 1].mean(dim=0).cpu().to(dtype=torch.float16).numpy()
        for j, ln in enumerate(lengths)
    ]

    return np.vstack(pooled)[rev_order]  # original order


# ────────────────────────── I/O helper functions ───────────────────────
def create_memmap(path: Path, shape: Tuple[int, ...], dtype: str) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"📝  Allocating memmap {path}  shape={shape}  dtype={dtype}")
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


def open_memmap(path: Path, shape: Tuple[int, ...], dtype: str, resume: bool) -> np.memmap:
    """Create new or open existing memmap depending on resume flag.

    Safety rules
    1. resume=True  & file exists  → open in r+
    2. resume=True  & file missing → error (cannot resume)
    3. resume=False & file missing → create new
    4. resume=False & file exists  → error (would overwrite)
    """
    if path.exists():
        if resume:
            log.info(f"📥  Resuming from existing memmap {path}")
            return np.memmap(path, mode="r+", dtype=dtype, shape=shape)
        log.error(f"{path} already exists. Delete it or use --resume to continue.")
        sys.exit(1)
    else:
        if resume:
            log.error(f"Expected existing file {path} for --resume but none found.")
            sys.exit(1)
        return create_memmap(path, shape, dtype)


def detect_rows(emb_mm: np.memmap, lab_mm: np.memmap) -> int:
    """Return count of rows already written by finding first all-zero embedding."""
    for i in range(emb_mm.shape[0]):
        if np.allclose(emb_mm[i], 0, atol=1e-7):
            return i
    return emb_mm.shape[0]  # All rows written



# ────────────────────────────── pipeline ───────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=str, default="data/TED/s90/s90_meta.parquet")
    p.add_argument("--seq",  type=str, default="data/TED/s90_reps.parquet",
                   help="Parquet/Arrow dataset with sequence_id + sequence")
    p.add_argument("--out",  type=str, default="data/TED/s90/embeddings/protT5")
    p.add_argument("--batch",        type=int, default=256,
                   help="GPU micro-batch size (push up to 384 on 80 GB A100)")
    p.add_argument("--arrow-batch",  type=int, default=64000,
                   help="Rows pulled per Arrow scan batch")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run using existing output files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    resume = args.resume

    # 1️⃣  load metadata once
    log.info(f"📥  Reading split metadata → {args.meta}")
    meta = pd.read_parquet(args.meta)
    required = {"sequence_id", "SF_label", "split"}
    if not required.issubset(meta.columns):
        log.error(f"Meta parquet must contain {required}")
        sys.exit(1)

    counts = meta["split"].value_counts().to_dict()        # {'train': …}
    splits = ("train", "val", "test")
    log.info("📊  Sequences per split: " + ", ".join(f"{s}:{counts.get(s,0):,}" for s in splits))

    # 2️⃣  pre-allocate memmaps
    emb_mm: Dict[str, np.memmap] = {}
    lab_mm: Dict[str, np.memmap] = {}
    write_ptr = {s: 0 for s in splits}

    for sp in splits:
        n = counts.get(sp, 0)
        emb_mm[sp] = open_memmap(out_dir / sp / "embeddings.npy", (n, 1024), "float16", resume)
        lab_mm[sp] = open_memmap(out_dir / sp / "labels.npy",     (n,),      "int32",  resume)

        if resume:
            write_ptr[sp] = detect_rows(emb_mm[sp], lab_mm[sp])

    already_done = sum(write_ptr.values())
    if resume and already_done:
        log.info(f"⏩  Resuming – {already_done:,} sequences already encoded")
    skipped_global = already_done

    #   index meta for O(1) look-up
    meta_idx = meta.set_index("sequence_id")[["split", "SF_label"]]

    # 3️⃣  load model
    model, tokenizer, device = load_prott5()

    # 4️⃣  stream sequences dataset
    log.info(f"🔄  Scanning sequence dataset → {args.seq}")
    ds_seq = ds.dataset(args.seq, format="parquet")

    total_seen = 0
    for rb in tqdm(ds_seq.to_batches(batch_size=args.arrow_batch,
                                     columns=["sequence_id", "sequence"]),
                   unit_scale=args.arrow_batch,
                   desc="Streaming"):
        ids = rb.column("sequence_id").to_pylist()
        seqs = rb.column("sequence").to_pylist()

        # Skip sequences that were processed in a previous run
        if resume and skipped_global:
            if skipped_global >= len(seqs):
                skipped_global -= len(seqs)
                total_seen += len(seqs)
                continue
            else:
                ids = ids[skipped_global:]
                seqs = seqs[skipped_global:]
                skipped_global = 0

        # look-up split + label
        meta_slice = meta_idx.loc[ids]  # preserves order
        splits_batch = meta_slice["split"].values
        labels_batch = meta_slice["SF_label"].values.astype("int32")

        # encode in GPU micro-batches
        for i in range(0, len(seqs), args.batch):
            sub_seqs  = seqs[i : i + args.batch]
            sub_split = splits_batch[i : i + args.batch]
            sub_label = labels_batch[i : i + args.batch]

            emb = encode_batch(model, tokenizer, device, sub_seqs)

            # write embeddings + labels
            for e, sp, lb in zip(emb, sub_split, sub_label):
                idx = write_ptr[sp]
                emb_mm[sp][idx] = e
                lab_mm[sp][idx] = lb
                write_ptr[sp] += 1

        total_seen += len(seqs)

        # flush periodically to be crash-safe
        if total_seen // 1_000_000 != (total_seen - len(seqs)) // 1_000_000:
            for sp in splits:
                emb_mm[sp].flush(); lab_mm[sp].flush()
            log.info(f"💾  Flushed after {total_seen:,} sequences")

    # 5️⃣  final consistency check + flush
    for sp in splits:
        expected = counts.get(sp, 0)
        assert write_ptr[sp] == expected, f"{sp}: wrote {write_ptr[sp]} vs expected {expected}"
        emb_mm[sp].flush(); lab_mm[sp].flush()

    log.info("✅  All done!")


if __name__ == "__main__":
    main()