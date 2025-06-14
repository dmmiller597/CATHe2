#!/usr/bin/env python3
# scripts/embeddings_s90_prott5.py
"""
One-pass, memory-efficient ProtT5 embedding pipeline for the 78 M-row S90 data set.

Outputs (per split)
───────────────────
<out-dir>/<split>/embeddings.npy   # mem-mapped  (N_split, 1024) float16
<out-dir>/<split>/labels.npy       # mem-mapped  (N_split,)     int32
<out-dir>/progress.json            # JSON dict with resume pointers

Design highlights
─────────────────
• <2 GB peak RAM   –  loads only the 4-column metadata table once.
• torch.compile()  –  JIT-compiles the model for a ~20-30% speed-up.
• Fast Tokenizer   –  Rust-based tokenizer for faster CPU pre-processing.
• Streaming Arrow  –  scans the 13 GB sequence parquet with constant memory.
• Robust resume    –  via a simple JSON file, no slow scanning of output files.
• Single write     –  no temporary objects, no post-processing.
• fp16 encoder     –  fits easily on a single A10 or A40.

Typical usage
─────────────
python scripts/embeddings_s90_prott5.py \
  --meta   data/TED/s90/s90_meta.parquet \
  --seq    data/TED/s90_reps.parquet \
  --out    data/TED/s90/embeddings/protT5 \
  --batch-tokens 80000 --arrow-batch 64000
"""

from __future__ import annotations
import argparse, logging, re, sys, json
from pathlib import Path
from typing import Dict, Tuple, List


import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from transformers import T5EncoderModel, T5TokenizerFast
from tqdm import tqdm

# ─────────────────────────────── logger ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("encode_s90")


# ─────────────────────────── model utilities ───────────────────────────
def load_prott5() -> Tuple[T5EncoderModel, T5TokenizerFast, torch.device]:
    """FP16 ProtT5 encoder ⇢ device, eval."""
    log.info("🔌 Loading ProtT5-XL-U50 (encoder-only, fp16)")
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    mdl = T5EncoderModel.from_pretrained(model_name)
    tok = T5TokenizerFast.from_pretrained(model_name, do_lower_case=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(dev).eval()  # fp16 weights are baked in

    # Guard against environments running PyTorch < 2.0
    if hasattr(torch, "compile"):
        log.info("🚀 Compiling model with torch.compile() for a speed-up")
        mdl = torch.compile(mdl)
    else:
        log.warning("torch.compile() not available – skipping JIT compilation")
    return mdl, tok, dev


aa_cleaner = re.compile(r"[UZOB]")  # ambiguous & rare → X


def preprocess(seq: str) -> str:
    """Replace rare AAs, add whitespace."""
    return " ".join(list(aa_cleaner.sub("X", seq)))


@torch.inference_mode()
def encode_batch(
    model: T5EncoderModel,
    tokenizer: T5TokenizerFast,
    device: torch.device,
    seqs: List[str],
) -> np.ndarray:
    """
    Encode a list of sequences → (B, 1024) float16 numpy.
    Keeps original order.
    """
    if not seqs:
        return np.empty((0, 1024), dtype=np.float16)

    # length-sort for speed, remember indices to restore order
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    rev_order = np.argsort(order)

    # Apply preprocessing (replace rare AAs, add whitespace) *before* tokenisation
    preproc_seqs = [preprocess(seqs[i]) for i in order]

    # The Rust-based fast tokenizer handles the whitespace and special tokens.
    tok_inputs = tokenizer(
        preproc_seqs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )

    inp_ids = tok_inputs["input_ids"].to(device)
    attn = tok_inputs["attention_mask"].to(device)

    hs = model(input_ids=inp_ids, attention_mask=attn).last_hidden_state

    # mean-pool (exclude <s> & </s>) per sequence
    lengths = attn.sum(dim=1)  # includes both special tokens
    mask = attn.bool()
    # Mask out the first and last token for each sequence
    mask[:, 0] = False
    mask.scatter_(1, lengths.unsqueeze(1) - 1, False)

    summed = (hs * mask.unsqueeze(-1)).sum(1)
    lens   = mask.sum(1).unsqueeze(-1)
    pooled = (summed / lens).to(dtype=torch.float16).cpu().numpy()

    return pooled[rev_order]  # original order


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


# ────────────────────────────── pipeline ───────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=str, default="data/TED/s90/s90_meta.parquet")
    p.add_argument("--seq",  type=str, default="data/TED/s90_reps.parquet",
                   help="Parquet/Arrow dataset with sequence_id + sequence")
    p.add_argument("--out",  type=str, default="data/TED/s90/embeddings/protT5")
    p.add_argument("--batch-tokens", type=int, default=80000,
                   help="Maximum tokens (<seq_len>+2) per micro-batch")
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

    # 2️⃣  set up outputs and resume state
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.json"

    write_ptr = {s: 0 for s in splits}
    if resume:
        if not progress_file.is_file():
            log.error(f"❌  --resume requires {progress_file}, but it was not found.")
            sys.exit(1)
        log.info(f" Reading resumption state from {progress_file}")
        with progress_file.open() as f:
            write_ptr = json.load(f)
    elif progress_file.is_file():
        log.error(f"❌  {progress_file} already exists. Delete it or use --resume.")
        sys.exit(1)

    emb_mm: Dict[str, np.memmap] = {}
    lab_mm: Dict[str, np.memmap] = {}
    for sp in splits:
        n = counts.get(sp, 0)
        emb_mm[sp] = open_memmap(out_dir / sp / "embeddings.npy", (n, 1024), "float16", resume)
        lab_mm[sp] = open_memmap(out_dir / sp / "labels.npy",     (n,),      "int32",  resume)

    already_done = sum(write_ptr.values())
    if resume and already_done:
        log.info(f"⏩  Resuming – {already_done:,} sequences already encoded")

    #   index meta for O(1) look-up
    meta_idx = meta.set_index("sequence_id")[["split", "SF_label"]]

    # 3️⃣  load model
    model, tokenizer, device = load_prott5()

    # 4️⃣  stream sequences dataset
    log.info(f"🔄  Scanning sequence dataset → {args.seq}")
    ds_seq = ds.dataset(args.seq, format="parquet")

    processed_since_flush = 0
    total_processed_in_run = 0

    # If resuming, skip the rows that have already been processed.
    scanner = ds_seq.scanner(
        batch_size=args.arrow_batch,
        columns=["sequence_id", "sequence"],
        offset=already_done
    )

    pbar_initial = already_done // args.arrow_batch if args.arrow_batch > 0 else 0
    pbar_total = (ds_seq.count_rows() + args.arrow_batch -1) // args.arrow_batch if args.arrow_batch > 0 else 1

    for rb in tqdm(scanner.to_batches(),
                   total=pbar_total, initial=pbar_initial,
                   unit_scale=args.arrow_batch,
                   desc="Streaming"):
        ids = rb.column("sequence_id").to_pylist()
        seqs = rb.column("sequence").to_pylist()

        # Look up metadata, safely skipping sequences not found in the meta file.
        meta_slice = meta_idx.reindex(ids)
        valid_mask = meta_slice["split"].notna()

        # If any sequences were missing from the metadata file, filter them out.
        if not valid_mask.all():
            n_missing = len(valid_mask) - valid_mask.sum()
            log.warning(f"Skipping {n_missing} sequences from arrow batch not found in meta file.")

            meta_slice = meta_slice[valid_mask]
            valid_mask_list = valid_mask.to_list()
            seqs = [s for (s, v) in zip(seqs, valid_mask_list) if v]

            if not seqs:
                continue

        splits_batch = meta_slice["split"].values
        labels_batch = meta_slice["SF_label"].values.astype("int32")

        # encode in GPU micro-batches governed by token budget rather than fixed count
        for sub_seqs, sub_split, sub_label in iter_token_batches(
                seqs, splits_batch, labels_batch, args.batch_tokens):

            emb = encode_batch(model, tokenizer, device, sub_seqs)

            # write embeddings + labels in batches per split
            for sp_val in splits:
                mask = (sub_split == sp_val)
                if not np.any(mask):
                    continue

                embs_to_write = emb[mask]
                labs_to_write = sub_label[mask]
                n_to_write = len(embs_to_write)

                start_idx = write_ptr[sp_val]
                end_idx = start_idx + n_to_write

                emb_mm[sp_val][start_idx:end_idx] = embs_to_write
                lab_mm[sp_val][start_idx:end_idx] = labs_to_write
                write_ptr[sp_val] = end_idx

        total_processed_in_run += len(seqs)
        processed_since_flush += len(seqs)

        # flush periodically to be crash-safe
        if processed_since_flush >= 1_000_000:
            for sp in splits:
                emb_mm[sp].flush(); lab_mm[sp].flush()
            with progress_file.open("w") as f:
                json.dump(write_ptr, f)
            log.info(f"💾  Flushed after {already_done + total_processed_in_run:,} total sequences")
            processed_since_flush = 0

    # 5️⃣  final consistency check + flush
    for sp in splits:
        expected = counts.get(sp, 0)
        assert write_ptr[sp] == expected, f"{sp}: wrote {write_ptr[sp]} vs expected {expected}"
        emb_mm[sp].flush(); lab_mm[sp].flush()
    with progress_file.open("w") as f:
        json.dump(write_ptr, f)

    log.info("✅  All done!")


# ─────────────────── variable-token batching helper ────────────────────

def iter_token_batches(
    seqs: List[str],
    splits: np.ndarray,
    labels: np.ndarray,
    token_budget: int,
):
    """Yield slices whose combined (len+2) ≤ token_budget.

    Keeps input order; the +2 reserves space for the <s> and </s> special tokens
    added by the tokenizer. The fast tokenizer does not require pre-tokenized inputs.
    """
    buf_seqs: List[str] = []
    buf_splits: List[str] = []
    buf_labels: List[int] = []
    running_tokens = 0

    for s, sp, lb in zip(seqs, splits, labels):
        t = len(s) + 2  # +2 for special tokens
        if buf_seqs and running_tokens + t > token_budget:
            yield buf_seqs, np.array(buf_splits), np.array(buf_labels)
            buf_seqs, buf_splits, buf_labels, running_tokens = [], [], [], 0
        buf_seqs.append(s)
        buf_splits.append(sp)
        buf_labels.append(lb)
        running_tokens += t

    if buf_seqs:
        yield buf_seqs, np.array(buf_splits), np.array(buf_labels)


if __name__ == "__main__":
    main()