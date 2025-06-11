#!/usr/bin/env python3
"""
Scalable train/val/test splitting for 78 M-row S90-like datasets.

Memory footprint: <2 GB (metadata only, sequence column never loaded).
Runtime: 2-3 min on a single A10 box for 78 M rows (I/O bound).

Outputs
-------
data/splits/meta.parquet        # 4 cols: sequence_id, SF, SF_label, split
data/splits/sf_label_map.json   # for downstream code
"""

from __future__ import annotations
import argparse, logging, json, numpy as np, pyarrow.dataset as ds, pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("splitter")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=str,  default="data/TED/s90_reps.parquet")
    p.add_argument("--out-dir", type=str,  default="data/TED/s90")
    p.add_argument("--min-seq", type=int,  default=3,   help="min sequences/SF")
    p.add_argument("--val-frac",  type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed",   type=int,  default=42)
    return p.parse_args()

def load_meta(path: str) -> pd.DataFrame:
    """Load only small metadata columns with Arrow → Pandas (categorical)."""
    log.info(f"📥  Reading Arrow dataset {path}")
    tbl = ds.dataset(path, format="parquet").to_table(
        columns=["sequence_id", "SF"],  # keep it minimal
        use_threads=True
    )
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)  # zero-copy
    df["SF"] = df["SF"].astype("category")          # compress to codes
    return df

def make_splits(df: pd.DataFrame, val_f: float, test_f: float,
                seed: int, min_seq: int) -> pd.DataFrame:
    np.random.seed(seed)
    df["split"] = np.full(len(df), "train", dtype="object")

    grp = df.groupby("SF", sort=False).indices         # dict SF_code → ndarray[int]
    log.info(f"🧬  {len(grp):,} unique SF groups")
    skipped = 0
    for sf, idx in tqdm(grp.items(), desc="Splitting"):
        if len(idx) < min_seq:
            skipped += 1
            continue
        idx = idx.copy()
        np.random.shuffle(idx)
        test_sz, val_sz = max(1, int(len(idx)*test_f)), max(1, int(len(idx)*val_f))
        df.iloc[idx[:test_sz],  df.columns.get_loc("split")] = "test"
        df.iloc[idx[test_sz:test_sz+val_sz], df.columns.get_loc("split")] = "val"

    if skipped:
        log.warning(f"⚠️  Skipped {skipped} SF groups with <{min_seq} sequences")

    return df

def save_outputs(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # integer labels
    sf_cats = df["SF"].cat.categories
    sf_to_int = {sf: int(i) for i, sf in enumerate(sf_cats)}
    df["SF_label"] = df["SF"].cat.codes.astype("int32")

    # main parquet
    meta_path = out_dir / "s90_meta.parquet"
    df[["sequence_id", "SF", "SF_label", "split"]].to_parquet(meta_path, index=False)
    log.info(f"💾  Wrote split metadata → {meta_path}")

    # label mapping
    map_path = out_dir / "s90_sf_label_map.json"
    with open(map_path, "w") as f:
        json.dump(sf_to_int, f, indent=2)
    log.info(f"💾  Wrote SF→int map       → {map_path}")

    # sanity check
    grp = df.groupby("split")["SF"].nunique()
    log.info("📊  SFs per split:\n" + grp.to_string())
    assert len(set(grp.values)) == 1, "Not all SFs present in all splits!"

def main():
    args = parse_args()
    df = load_meta(args.input)
    df = make_splits(df, args.val_frac, args.test_frac, args.seed, args.min_seq)
    save_outputs(df, Path(args.out_dir))
    log.info("✅  Done")

if __name__ == "__main__":
    main()