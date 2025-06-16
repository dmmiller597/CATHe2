#!/usr/bin/env python3
"""Efficient ProtT5 mean-pool embeddings written directly to a memory-mapped array.

Usage example:
    python -m cathe2.scripts.embed_prott5 \
        --fasta data/TED/s30/s30_full.fasta \
        --meta  data/TED/s30/s30_full.json \
        --out-dir data/TED/s30/protT5 \
        --device cuda:0 \
        --max-tokens 150000 

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
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

_LOGGER = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# FASTA & JSON helpers (stand-alone – no external utils)
# ────────────────────────────────────────────────────────────────────────────────

def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    """Parse a (multi-)FASTA file.

    Returns two parallel lists: *(ids, sequences)*.  The *id* is the full header
    line minus the leading ">" and everything after the first whitespace.
    """
    ids: List[str] = []
    seqs: List[str] = []
    seq_buf: List[str] = []

    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush previous record
                if seq_buf:
                    seqs.append("".join(seq_buf))
                    seq_buf.clear()
                ids.append(line[1:].split()[0])
            else:
                seq_buf.append(line)
        # final record
        if seq_buf:
            seqs.append("".join(seq_buf))

    if len(ids) != len(seqs):
        raise ValueError(f"FASTA parsing produced {len(ids)} ids but {len(seqs)} sequences")
    return ids, seqs


def load_json_meta(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)

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


def embed_sequences(
    ids: List[str],
    seqs: List[str],
    memmap: np.memmap,
    tokenizer: AutoTokenizer,
    model: T5EncoderModel,
    device: torch.device,
    max_tokens: int,
):
    """Encode and write each protein's mean embedding into *memmap* in-place."""

    # 1️⃣ order sequences by length (descending) for better packing
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    rev_order = np.argsort(order)

    seqs_sorted = [seqs[i] for i in order]

    # 2️⃣ clean & whitespace-tokenise
    seqs_tok = [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in seqs_sorted]

    _LOGGER.info("Embedding %d sequences", len(seqs_tok))
    with torch.inference_mode():
        pbar = tqdm(total=len(seqs_tok), unit="seq")
        for batch_idx, (row_idxs, batch_seqs) in enumerate(
            _iter_token_batches(seqs_tok, order, max_tokens)
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
                memmap[row_idxs[j]] = mean_vec.cpu().numpy().astype(np.float32)

            pbar.update(len(batch_seqs))
        pbar.close()

    # 3️⃣ reorder back to original sequence order
    memmap[:] = memmap[rev_order]

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
    p.add_argument("--fasta", required=True, default="data/TED/s30/s30_full.fasta", type=Path, help="Input FASTA file")
    p.add_argument("--meta", required=True, default="data/TED/s30/s30_full.json", type=Path, help="JSON with SF & split labels")
    p.add_argument("--out-dir", required=True, default="data/TED/s30/protT5", type=Path, help="Directory for outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-tokens", type=int, default=150000, help="Token budget per batch")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)

    _LOGGER.info("Reading FASTA from %s", args.fasta)
    ids, seqs = read_fasta(args.fasta)

    meta = load_json_meta(args.meta)

    if len(ids) == 0:
        _LOGGER.error("No sequences found in FASTA – aborting")
        sys.exit(1)

    # ── allocate mem-map ───────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = args.out_dir / "protT5_embeddings.npy"
    mmap_arr = np.memmap(mmap_path, mode="w+", dtype="float32", shape=(len(ids), 1024))

    _LOGGER.info("Initialising ProtT5 (this can take a minute)…")
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model.eval().to(device)

    embed_sequences(ids, seqs, mmap_arr, tokenizer, model, device, args.max_tokens)

    mmap_arr.flush()  # ensure everything hits disk

    # ── auxiliary artefacts ────────────────────────────────────────────────────
    save_json({"id2idx": {id_: i for i, id_ in enumerate(ids)}}, args.out_dir / "protT5_id_index_map.json")

    write_split_files(meta, ids, args.out_dir.parent / "splits")

    _LOGGER.info("Finished. Mem-map stored at %s", mmap_path)


if __name__ == "__main__":
    main() 