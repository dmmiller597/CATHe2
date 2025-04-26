# create_ted_dataset_split.py

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── GLOBAL DEFAULTS ───────────────────────────────────────────────────────────
DEFAULT_TEST_SIZE = 300
DEFAULT_VAL_SIZE = 200
DEFAULT_PIDE_FILTER = 0.20
DEFAULT_PIDE_CLUSTER = 0.95
DEFAULT_COV_CLUSTER = 0.95
DEFAULT_FAMILY_THRESHOLD = 100
DEFAULT_MMSEQS_ITERS = 3
DEFAULT_MMSEQS_SENS = 7.5
DEFAULT_MMSEQS_COV_MODE = 0
DEFAULT_RANDOM_SEED = 42
DEFAULT_THREADS = 15 # Sensible default, adjust based on node cores

# ─── UTIL: RUN SHELL COMMAND ────────────────────────────────────────────────────
def run_cmd(cmd: List[str]) -> None:
    """Run command, log stdout/stderr, exit on error."""
    logging.debug(f"> {' '.join(cmd)}")
    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.debug(cp.stdout)
        logging.debug(cp.stderr)
    except FileNotFoundError:
        logging.error(f"Executable not found: {cmd[0]}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed [{e.returncode}]: {' '.join(e.cmd)}")
        logging.error(e.stderr or e.stdout)
        sys.exit(1)

# ─── UTIL: FASTA I/O ────────────────────────────────────────────────────────────
def df_to_fasta(df: pd.DataFrame, fasta: Path, id_col: str, seq_col: str) -> None:
    """Write DataFrame to FASTA (80-char wrap)."""
    logging.info(f"Writing {len(df)} sequences to {fasta}")
    with fasta.open("w") as out:
        for sid, seq in zip(df[id_col], df[seq_col]):
            out.write(f">{sid}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")

# ─── UTIL: MMSEQS2 DB CREATION ──────────────────────────────────────────────────
def create_db(fasta: Path, db_prefix: Path, mmseqs: str) -> None:
    """mmseqs createdb fasta → db_prefix."""
    run_cmd([mmseqs, "createdb", str(fasta), str(db_prefix)])

# ─── UTIL: MMSEQS2 SEARCH ───────────────────────────────────────────────────────
def mmseqs_search(
    query_db: Path, target_db: Path, result_db: Path, result_tsv: Path,
    tmp_dir: Path, mmseqs: str, sens: float, iters: int,
    cov_mode: int, min_seq_id: float, threads: int
) -> None:
    """Run mmseqs search + createtsv → produces result_tsv."""
    run_cmd([
        mmseqs, "search",
        str(query_db), str(target_db),
        str(result_db), str(tmp_dir),
        "-s", str(sens),
        "--num-iterations", str(iters),
        "--cov-mode", str(cov_mode),
        "-c", str(0),
        "--min-seq-id", str(min_seq_id),
        "--threads", str(threads)
    ])
    run_cmd([
        mmseqs, "createtsv",
        str(query_db), str(target_db),
        str(result_db), str(result_tsv)
    ])

# ─── UTIL: PARSE MMSEQS TSV ────────────────────────────────────────────────────
def parse_search_tsv(tsv: Path) -> pd.DataFrame:
    cols = [
        "query", "target", "pide", "alnlen", "mismatch",
        "gapopen", "qstart", "qend", "tstart", "tend",
        "evalue", "bits"
    ]
    try:
        return pd.read_csv(tsv, sep="\t", header=None, names=cols)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)

def extract_ids_above(df: pd.DataFrame, threshold: float) -> Set[str]:
    """Return set of query IDs with PIDE > threshold."""
    return set(df.loc[df["pide"] > threshold, "query"].unique())

# ─── UTIL: MMSEQS2 CLUSTERING ──────────────────────────────────────────────────
def mmseqs_cluster(
    db: Path, clu_db: Path, tmp_dir: Path,
    cluster_tsv: Path, mmseqs: str, min_seq_id: float,
    cov_mode: int, cov_thresh: float, threads: int
) -> None:
    run_cmd([
        mmseqs, "cluster",
        str(db), str(clu_db), str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "--cov-mode", str(cov_mode),
        "-c", str(cov_thresh),
        "--threads", str(threads)
    ])
    run_cmd([
        mmseqs, "createtsv",
        str(db), str(db), str(clu_db), str(cluster_tsv)
    ])

def extract_representatives(cluster_tsv: Path) -> Set[str]:
    try:
        df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
        return set(df["rep"].unique())
    except pd.errors.EmptyDataError:
        return set()

# ─── SPLIT LOGIC ───────────────────────────────────────────────────────────────
def split_test_val(
    df: pd.DataFrame, test_n: int, val_n: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly select test/val ensuring disjoint superfamilies."""
    rnd = np.random.RandomState(seed)
    # 1) sample test
    all_idx = np.arange(len(df))
    test_idx = rnd.choice(all_idx, test_n, replace=False)
    test_df = df.iloc[test_idx]
    remaining = df.drop(index=test_df.index)
    test_sfs = set(test_df["SF"].unique())
    # 2) sample val respecting SF
    val_idx = []
    for idx in rnd.permutation(remaining.index):
        if len(val_idx) >= val_n:
            break
        if remaining.at[idx, "SF"] not in test_sfs:
            val_idx.append(idx)
    val_df = remaining.loc[val_idx]
    train_pool = remaining.drop(index=val_df.index)
    logging.info(f"Test {len(test_df)}, Val {len(val_df)}, TrainPool {len(train_pool)}")
    return test_df, val_df, train_pool

def filter_training_pool(
    train_pool: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame,
    tmp_dir: Path, mmseqs: str, sens: float, iters: int,
    cov_mode: int, pid_thresh: float, threads: int
) -> pd.DataFrame:
    """Remove train_pool sequences > pid_thresh to test/val."""
    # write FASTA & db
    fa_pool = tmp_dir/"train_pool.fasta"; df_to_fasta(train_pool, fa_pool, "sequence_id", "sequence")
    db_pool = tmp_dir/"trainDB"; create_db(fa_pool, db_pool, mmseqs)

    fa_test = tmp_dir/"test.fasta"; df_to_fasta(test_df, fa_test, "sequence_id", "sequence")
    db_test = tmp_dir/"testDB"; create_db(fa_test, db_test, mmseqs)

    fa_val = tmp_dir/"val.fasta"; df_to_fasta(val_df, fa_val, "sequence_id", "sequence")
    db_val = tmp_dir/"valDB"; create_db(fa_val, db_val, mmseqs)

    # search pool vs test
    tsv1 = tmp_dir/"pool_vs_test.tsv"; resdb1 = tmp_dir/"res1"
    mmseqs_search(db_pool, db_test, resdb1, tsv1, tmp_dir, mmseqs, sens, iters, cov_mode, pid_thresh, threads)
    ids1 = extract_ids_above(parse_search_tsv(tsv1), pid_thresh)
    # search pool vs val
    tsv2 = tmp_dir/"pool_vs_val.tsv"; resdb2 = tmp_dir/"res2"
    mmseqs_search(db_pool, db_val, resdb2, tsv2, tmp_dir, mmseqs, sens, iters, cov_mode, pid_thresh, threads)
    ids2 = extract_ids_above(parse_search_tsv(tsv2), pid_thresh)

    bad = ids1.union(ids2)
    filtered = train_pool[~train_pool["sequence_id"].isin(bad)]
    logging.info(f"Filtered train pool: {len(filtered)} left ({len(bad)} removed)")
    return filtered

def cluster_large_families(
    df: pd.DataFrame, fam_thresh: int, tmp_dir: Path, mmseqs: str,
    pid_cluster: float, cov_mode: int, cov_cluster: float, threads: int
) -> pd.DataFrame:
    """For SF groups > fam_thresh, cluster and keep representatives."""
    reps = []
    for sf, group in tqdm(df.groupby("SF"), desc="Clustering families"):
        if len(group) <= fam_thresh:
            reps.extend(group["sequence_id"].tolist())
        else:
            # write & cluster
            fam_fa = tmp_dir/f"fam_{sf}.fasta"; df_to_fasta(group, fam_fa, "sequence_id", "sequence")
            fam_db = tmp_dir/f"famDB_{sf}"; create_db(fam_fa, fam_db, mmseqs)
            clu_db = tmp_dir/f"cluDB_{sf}"; clu_tsv = tmp_dir/f"clu_{sf}.tsv"
            mmseqs_cluster(fam_db, clu_db, tmp_dir, clu_tsv, mmseqs, pid_cluster, cov_mode, cov_cluster, threads)
            reps_sf = extract_representatives(clu_tsv)
            reps.extend(reps_sf)
    final = df[df["sequence_id"].isin(reps)].copy()
    logging.info(f"After clustering: {len(final)} training sequences")
    return final

def build_lookup_eval(
    all_df: pd.DataFrame, test_df: pd.DataFrame, tmp_dir: Path,
    mmseqs: str, sens: float, iters: int, cov_mode: int, pid_thresh: float, threads: int
) -> pd.DataFrame:
    """Filter all_df–test by identity to test, ignore val."""
    pool = all_df[~all_df["sequence_id"].isin(test_df["sequence_id"])]
    return filter_training_pool(pool, test_df, pd.DataFrame(columns=all_df.columns),
                                tmp_dir, mmseqs, sens, iters, cov_mode, pid_thresh, threads)

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Create train/val/test splits for TED dataset."
    )
    p.add_argument("input_parquet", type=Path)
    p.add_argument("-o", "--output_dir", type=Path, required=True)
    p.add_argument("--mmseqs", default="mmseqs", help="MMSeqs2 binary")
    p.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE)
    p.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    p.add_argument("--pide_filter", type=float, default=DEFAULT_PIDE_FILTER)
    p.add_argument("--pide_cluster", type=float, default=DEFAULT_PIDE_CLUSTER)
    p.add_argument("--cov_cluster", type=float, default=DEFAULT_COV_CLUSTER)
    p.add_argument("--family_thresh", type=int, default=DEFAULT_FAMILY_THRESHOLD)
    p.add_argument("--iters", type=int, default=DEFAULT_MMSEQS_ITERS)
    p.add_argument("--sens", type=float, default=DEFAULT_MMSEQS_SENS)
    p.add_argument("--cov_mode", type=int, default=DEFAULT_MMSEQS_COV_MODE)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help="Number of CPU threads for MMSeqs2")
    args = p.parse_args()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO
    )

    # Validate input
    if not args.input_parquet.is_file():
        logging.error("Input parquet not found.")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading input parquet...")
    df = pd.read_parquet(args.input_parquet)
    for col in ("sequence_id","sequence","length","SF"):
        if col not in df.columns:
            logging.error(f"Missing column: {col}")
            sys.exit(1)
    # sanitize IDs
    df["sequence_id"] = df["sequence_id"].astype(str).str.replace(r"[^A-Za-z0-9_.-]", "_", regex=True)

    # Work in temp dir
    with tempfile.TemporaryDirectory(prefix="ted_split_") as tmpd:
        tmp_dir = Path(tmpd)
        logging.info(f"Using temporary directory: {tmp_dir}")

        # 1) split test & val
        test_df, val_df, train_pool = split_test_val(
            df, args.test_size, args.val_size, args.seed
        )

        # 2) remove >20% identity to test/val
        filtered_train = filter_training_pool(
            train_pool, test_df, val_df, tmp_dir,
            args.mmseqs, args.sens, args.iters, args.cov_mode, args.pide_filter,
            args.threads
        )

        # 3) cluster large families
        train_df = cluster_large_families(
            filtered_train, args.family_thresh, tmp_dir,
            args.mmseqs, args.pide_cluster, args.cov_mode, args.cov_cluster,
            args.threads
        )

        # 4) lookup for final eval
        lookup_df = build_lookup_eval(
            df, test_df, tmp_dir,
            args.mmseqs, args.sens, args.iters, args.cov_mode, args.pide_filter,
            args.threads
        )

        # 5) save outputs
        logging.info("Saving outputs...")
        test_df.to_parquet(args.output_dir/"test_set.parquet", index=False)
        val_df.to_parquet(args.output_dir/"validation_set.parquet", index=False)
        train_df.to_parquet(args.output_dir/"training_set.parquet", index=False)
        lookup_df.to_parquet(args.output_dir/"evaluation_lookup_set.parquet", index=False)

    logging.info("Done.")

if __name__ == "__main__":
    main()