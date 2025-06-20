import os
import subprocess
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_fasta(file_path: Path) -> Dict[str, str]:
    """Parses a FASTA file into a dictionary of headers to sequences."""
    sequences = {}
    current_header = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_header = line[1:]
                sequences[current_header] = ''
            elif current_header:
                sequences[current_header] += line
    return sequences

def write_fasta(file_path: Path, sequences: dict[str, str]):
    """Writes a dictionary of sequences to a FASTA file."""
    with open(file_path, 'w') as f:
        for header, seq in sequences.items():
            f.write(f'>{header}\n{seq}\n')

def get_cath_label(header: str) -> str:
    """Extracts the CATH superfamily label from a FASTA header."""
    try:
        # Expected format: >cath|4_4_0|cath_id/sequence_indices|SF
        return header.split('|')[-1]
    except IndexError:
        logging.warning(f"Could not parse CATH label from header: {header}")
        return ""

def run_mmseqs2(
    input_file: Path, 
    output_dir: Path, 
    identity: float, 
    sensitivity: float, 
    coverage: float = 0.8
) -> Path:
    """Runs MMSeqs2 easy-cluster to cluster sequences."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_file_base = output_dir / f"cluster_{int(identity*100)}"
    tmp_dir = output_dir / "tmp"

    # Resolve input_file to an absolute path to ensure it's found
    # when the subprocess is run with a different working directory.
    cmd = [
        "mmseqs", "easy-cluster", str(input_file.resolve()), str(cluster_file_base), str(tmp_dir),
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", "0",
        "-s", str(sensitivity),
        "--threads", str(os.cpu_count() or 1)
    ]
    
    logging.info(f"Running MMSeqs2 for {int(identity*100)}% identity...")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=output_dir)
        logging.info("MMSeqs2 completed successfully.")
        logging.debug(f"MMSeqs2 stdout:\n{result.stdout}")
        logging.debug(f"MMSeqs2 stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"MMSeqs2 failed with exit code {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        raise
    
    cluster_tsv = cluster_file_base.with_name(f"{cluster_file_base.name}_cluster.tsv")
    if not cluster_tsv.is_file():
        raise FileNotFoundError(f"MMSeqs2 did not produce the expected cluster file: {cluster_tsv}")
        
    return cluster_tsv

def parse_clusters(cluster_file: Path) -> Dict[str, List[str]]:
    """Parses the MMSeqs2 cluster file."""
    clusters = defaultdict(list)
    with open(cluster_file, 'r') as f:
        for line in f:
            rep, member = line.strip().split('\t')
            clusters[rep].append(member)
    return dict(clusters)

def split_representatives(
    clusters: Dict[str, List[str]], 
    test_ratio: float, 
    val_ratio: float,
    min_label_count: int,
    random_state: int = 42
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Splits cluster representatives into train, validation, and test sets.
    The split is stratified by CATH label to maintain label distribution.
    Labels with fewer than a minimum number of representatives are moved entirely to the training set.
    """
    representatives = list(clusters.keys())
    
    # Create a mapping of representative to its label for stratification
    rep_to_label = {rep: get_cath_label(rep) for rep in representatives}
    
    # Count occurrences of each label to identify rare labels
    label_counts = Counter(rep_to_label.values())
        
    # Representatives with rare labels must go into the training set to avoid issues with stratification
    # and to ensure the model sees these labels during training.
    must_train_reps = {rep for rep, label in rep_to_label.items() if label_counts[label] < min_label_count}
    
    # Representatives that are eligible for splitting have more common labels
    splittable_reps = [rep for rep in representatives if rep not in must_train_reps]
    splittable_labels = [rep_to_label[rep] for rep in splittable_reps]

    if not splittable_reps:
        logging.warning(f"No representatives available for splitting after filtering for rare labels (min count: {min_label_count}).")
        return must_train_reps, set(), set()

    # Split splittable_reps into a temporary training set and a test set
    try:
        train_val_reps, test_reps = train_test_split(
            splittable_reps,
            test_size=test_ratio,
            stratify=splittable_labels,
            random_state=random_state
        )
    except ValueError:
        # This can happen if a label has only one member in the splittable set, making stratification impossible.
        logging.warning("Stratified test split failed. Falling back to unstratified split.")
        train_val_reps, test_reps = train_test_split(splittable_reps, test_size=test_ratio, random_state=random_state)

    # Adjust validation ratio for the second split from the remaining data
    if 1.0 - test_ratio <= 1e-9: # Avoid division by zero with float comparison
        val_ratio_adj = 0
    else:
        val_ratio_adj = val_ratio / (1.0 - test_ratio)

    if val_ratio_adj > 0 and len(train_val_reps) > 1:
        train_val_labels = [rep_to_label[rep] for rep in train_val_reps]
        # Split the temporary training set into the final training and validation sets
        try:
            train_reps_final, val_reps = train_test_split(
                train_val_reps,
                test_size=val_ratio_adj,
                stratify=train_val_labels,
                random_state=random_state
            )
        except ValueError:
            logging.warning("Stratified validation split failed. Falling back to unstratified split.")
            train_reps_final, val_reps = train_test_split(train_val_reps, test_size=val_ratio_adj, random_state=random_state)
    else:
        # Not enough representatives to create a validation set or val_ratio is 0
        train_reps_final, val_reps = train_val_reps, []

    # The final training set is the union of the must-train reps and the split training reps
    final_train_reps = must_train_reps.union(set(train_reps_final))

    return final_train_reps, set(val_reps), set(test_reps)

def enforce_label_consistency(
    train_seqs: Dict[str, str], 
    val_seqs: Dict[str, str], 
    test_seqs: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Ensures that validation and test sets only contain labels present in the training set.
    Any sequence in the validation or test set with a label not found in the training
    set is removed to prevent evaluation on unseen classes.
    """
    train_labels = {get_cath_label(h) for h in train_seqs.keys()}
    
    val_seqs_filtered = {h: s for h, s in val_seqs.items() if get_cath_label(h) in train_labels}
    test_seqs_filtered = {h: s for h, s in test_seqs.items() if get_cath_label(h) in train_labels}
    
    val_removed = len(val_seqs) - len(val_seqs_filtered)
    test_removed = len(test_seqs) - len(test_seqs_filtered)
    
    if val_removed > 0:
        logging.info(f"Removed {val_removed} sequences from validation set due to missing labels in training set.")
    if test_removed > 0:
        logging.info(f"Removed {test_removed} sequences from test set due to missing labels in training set.")
        
    return val_seqs_filtered, test_seqs_filtered

def process_identity_level(
    identity_percent: int,
    base_dir: Path,
    all_sequences: Dict[str, str],
    input_fasta: Path,
    args: argparse.Namespace
) -> Optional[Dict]:
    """
    Runs the full clustering and splitting pipeline for a single identity threshold.

    Args:
        identity_percent: The sequence identity threshold (e.g., 30 for 30%).
        base_dir: The main output directory for all datasets.
        all_sequences: A dictionary of all sequences from the input FASTA.
        input_fasta: Path to the source FASTA file.
        args: The command-line arguments.

    Returns:
        A dictionary containing the summary statistics for this identity level, or None if processing fails.
    """
    identity_threshold = identity_percent / 100.0
    identity_dir = base_dir / f"s{identity_percent}"
    identity_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = args.high_sensitivity if identity_threshold < args.sensitivity_threshold else args.low_sensitivity
    
    logging.info(f"\n--- Processing {identity_percent}% Identity Threshold ---")
    
    try:
        cluster_file = run_mmseqs2(
            input_fasta, 
            identity_dir, 
            identity_threshold, 
            sensitivity,
            args.coverage
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Failed to generate clusters for {identity_percent}% identity. Skipping. Error: {e}")
        return None

    clusters = parse_clusters(cluster_file)
    logging.info(f"Found {len(clusters)} clusters at {identity_percent}% identity.")

    train_reps, val_reps, test_reps = split_representatives(
        clusters, 
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        min_label_count=args.min_label_count_for_split,
        random_state=args.random_state
    )

    # Training set includes all members of training clusters
    train_seqs = {}
    for rep in train_reps:
        # The representative is part of its own cluster list in this implementation
        for member in clusters.get(rep, [rep]):
            if member in all_sequences:
                train_seqs[member] = all_sequences[member]

    # Validation and test sets contain ONLY the representatives
    val_seqs = {rep: all_sequences[rep] for rep in val_reps if rep in all_sequences}
    test_seqs = {rep: all_sequences[rep] for rep in test_reps if rep in all_sequences}
    
    logging.info(f"Initial split sizes: Train reps={len(train_reps)}, Val reps={len(val_reps)}, Test reps={len(test_reps)}")

    # Enforce that validation and test labels are present in the training set
    val_seqs, test_seqs = enforce_label_consistency(train_seqs, val_seqs, test_seqs)

    logging.info(f"Final split sizes: Train={len(train_seqs)}, Validation={len(val_seqs)}, Test={len(test_seqs)}")

    # Save the filtered datasets
    write_fasta(identity_dir / 'train.fasta', train_seqs)
    write_fasta(identity_dir / 'valid.fasta', val_seqs)
    write_fasta(identity_dir / 'test.fasta', test_seqs)

    # Record results for summary table
    train_labels = {get_cath_label(h) for h in train_seqs}
    val_labels = {get_cath_label(h) for h in val_seqs}
    test_labels = {get_cath_label(h) for h in test_seqs}
    
    return {
        "Identity": f"{identity_percent}%",
        "Test Seqs": len(test_seqs),
        "Test Labels": len(test_labels),
        "Val Seqs": len(val_seqs),
        "Val Labels": len(val_labels),
        "Train Seqs": len(train_seqs),
        "Train Labels": len(train_labels),
    }

def main(args):
    input_fasta = Path(args.input_fasta)
    if not input_fasta.is_file():
        logging.error(f"Input FASTA file not found at: {input_fasta}")
        return

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logging.info("--- Script Configuration ---")
    for key, value in vars(args).items():
        logging.info(f"{key:<25}: {value}")
    logging.info("----------------------------")

    logging.info("Parsing all sequences from input FASTA...")
    all_sequences = parse_fasta(input_fasta)
    logging.info(f"Found {len(all_sequences)} total sequences.")

    results_summary = []
    
    identities = [int(i) for i in args.identity_thresholds.split(',')]

    for identity_percent in identities:
        result = process_identity_level(
            identity_percent,
            output_base_dir,
            all_sequences,
            input_fasta,
            args
        )
        if result:
            results_summary.append(result)

    # Print summary table
    print("\n" + "="*90)
    print(" " * 35 + "Dataset Creation Summary")
    print("="*90)
    if results_summary:
        headers = results_summary[0].keys()
        # Calculate column widths for neat printing
        col_widths = {h: max(len(h), max((len(str(r[h])) for r in results_summary), default=0)) for h in headers}
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        print(header_line)
        print("-" * len(header_line))
        for result in results_summary:
            row_line = " | ".join(str(result[h]).ljust(col_widths[h]) for h in headers)
            print(row_line)
    else:
        print("No datasets were generated.")
    print("="*90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cluster CATH sequences with MMSeqs2 and create train/validation/test splits at multiple identity thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_fasta", 
        type=str, 
        default="data/CATH/CATH_S100_with_SF.fasta",
        help="Path to the input FASTA file with CATH superfamily labels in the header."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/CATH/clustered_datasets",
        help="Base directory to save the clustered dataset files."
    )
    parser.add_argument(
        "--identity_thresholds",
        type=str,
        default="10,20,30,40,50,60,70,80,90",
        help="Comma-separated list of identity percentages to process (e.g., '30,50,70')."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of clusters to use for the test set."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of clusters to use for the validation set (relative to the whole set)."
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Minimum coverage for MMSeqs2 clustering ('-c' parameter)."
    )
    parser.add_argument(
        "--sensitivity_threshold",
        type=float,
        default=0.5,
        help="Sequence identity threshold below which high sensitivity MMSeqs2 settings are used."
    )
    parser.add_argument(
        "--high_sensitivity",
        type=float,
        default=7.5,
        help="Sensitivity for identities below the sensitivity threshold."
    )
    parser.add_argument(
        "--low_sensitivity",
        type=float,
        default=4.0,
        help="Sensitivity for identities at or above the sensitivity threshold."
    )
    parser.add_argument(
        "--min_label_count_for_split",
        type=int,
        default=3,
        help="Minimum number of representatives for a CATH label to be included in test/validation splits."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility of splits."
    )
    
    args = parser.parse_args()
    main(args) 