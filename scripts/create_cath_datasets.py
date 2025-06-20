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

    # Define simple, relative paths for MMSeqs2 to use inside its working directory
    cluster_file_base_name = f"cluster_{int(identity*100)}"
    tmp_dir_name = "tmp"

    # Resolve input_file to an absolute path to ensure it's found by the subprocess.
    # The output and tmp paths are relative, as they are used within the `cwd`.
    cmd = [
        "mmseqs", "easy-cluster", str(input_file.resolve()), 
        cluster_file_base_name, 
        tmp_dir_name,
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", "0",
        "-s", str(sensitivity),
        "--threads", str(os.cpu_count() or 1)
    ]
    
    logging.info(f"Running MMSeqs2 for {int(identity*100)}% identity...")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command from within the specified output directory.
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=output_dir)
        logging.info("MMSeqs2 completed successfully.")
        logging.debug(f"MMSeqs2 stdout:\n{result.stdout}")
        logging.debug(f"MMSeqs2 stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"MMSeqs2 failed with exit code {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        raise
    
    # The output cluster file will be inside the output_dir.
    cluster_tsv = output_dir / f"{cluster_file_base_name}_cluster.tsv"
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
    if test_ratio > 0:
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
    else:
        train_val_reps = splittable_reps
        test_reps = []

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

def generate_summary(
    final_training_set: Dict[str, str],
    final_validation_set: Dict[str, str],
    final_test_sets: Dict[int, Dict[str, str]]
) -> str:
    """Generates a formatted text summary of the created datasets."""
    
    final_master_train_labels = {get_cath_label(h) for h in final_training_set.keys()}
    final_master_val_labels = {get_cath_label(h) for h in final_validation_set.keys()}

    summary_data = []
    for identity, test_seqs in sorted(final_test_sets.items()):
        final_test_labels = {get_cath_label(h) for h in test_seqs.keys()}
        summary_data.append({
            "Identity": f"S{identity}",
            "Test Seqs": len(test_seqs),
            "Test SFs": len(final_test_labels),
        })

    lines = []
    lines.append("="*90)
    lines.append(" " * 30 + "Final Dataset Creation Summary")
    lines.append("="*90)
    lines.append(f"Final Training Set:     {len(final_training_set):>7} sequences, {len(final_master_train_labels):>5} SFs")
    lines.append(f"Final Validation Set:   {len(final_validation_set):>7} sequences, {len(final_master_val_labels):>5} SFs")
    lines.append("-" * 90)

    if summary_data:
        headers = summary_data[0].keys()
        col_widths = {h: max(len(h), max((len(str(r[h])) for r in summary_data), default=0)) for h in headers}
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        for result in summary_data:
            row_line = " | ".join(str(result[h]).ljust(col_widths[h]) for h in headers)
            lines.append(row_line)
    else:
        lines.append("No test sets were generated.")
    lines.append("="*90)
    
    return "\n".join(lines)

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

    # --- Revised Logic for Dataset Creation ---
    # Strategy:
    # 1. Hold out a validation set first. All members of validation clusters are removed from the main pool.
    # 2. Iteratively build test sets from highest identity (90) to lowest (10).
    # 3. In each iteration, cluster the current pool, identify test clusters, and add their
    #    representatives to a cumulative test set. Then, remove ALL MEMBERS of these test clusters
    #    from the pool to prevent data leakage in subsequent, lower-identity steps.
    # 4. The final remaining pool becomes the training set.
    # 5. Perform a final label consistency check across all sets.

    # 1. Create a global validation set to hold out from the start.
    logging.info("\n--- Step 1: Creating Global Validation Set ---")
    val_identity_thresh = 0.5  # A medium identity for a reasonable initial split
    val_dir = output_base_dir / "validation_clustering"
    try:
        val_cluster_file = run_mmseqs2(input_fasta, val_dir, val_identity_thresh, args.low_sensitivity, args.coverage)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Failed to generate initial validation clusters. Aborting. Error: {e}")
        return

    all_clusters = parse_clusters(val_cluster_file)
    
    # Split reps to decide which clusters go to validation vs. the main pool
    reps_for_pool, val_reps, _ = split_representatives(
        all_clusters,
        test_ratio=0.0,  # No test set at this stage
        val_ratio=args.val_ratio,
        min_label_count=args.min_label_count_for_split,
        random_state=args.random_state
    )

    # The validation set contains ALL members of the validation clusters.
    final_validation_set = {}
    for rep in val_reps:
        for member in all_clusters.get(rep, [rep]):
            if member in all_sequences:
                final_validation_set[member] = all_sequences[member]

    # The initial pool for training/test creation includes all members of non-validation clusters.
    current_pool_sequences = {}
    for rep in reps_for_pool:
        for member in all_clusters.get(rep, [rep]):
            if member in all_sequences:
                current_pool_sequences[member] = all_sequences[member]

    logging.info(f"Held out {len(final_validation_set)} sequences for the global validation set.")
    logging.info(f"Initial pool for training/test creation: {len(current_pool_sequences)} sequences.")

    # 2. Iteratively create test sets and shrink the training pool
    final_test_sets = {}
    identities = sorted([int(i) for i in args.identity_thresholds.split(',')], reverse=True)
    
    # This set will hold all representatives that have been assigned to any test set.
    cumulative_test_reps_headers = set()

    for identity_percent in identities:
        logging.info(f"\n--- Step 2: Processing {identity_percent}% Identity Level ---")
        
        if not current_pool_sequences:
            logging.warning(f"Training pool is empty. Stopping at {identity_percent}%.")
            break

        # Create a temporary fasta file for the current (shrinking) pool
        pool_fasta_path = output_base_dir / f"temp_pool_{identity_percent}.fasta"
        write_fasta(pool_fasta_path, current_pool_sequences)

        identity_threshold = identity_percent / 100.0
        identity_dir = output_base_dir / f"s{identity_percent}"
        sensitivity = args.high_sensitivity if identity_threshold < args.sensitivity_threshold else args.low_sensitivity

        try:
            cluster_file = run_mmseqs2(
                pool_fasta_path,
                identity_dir,
                identity_threshold,
                sensitivity,
                args.coverage
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"Failed to generate clusters for {identity_percent}%. Skipping. Error: {e}")
            if pool_fasta_path.exists():
                pool_fasta_path.unlink()
            continue
        
        clusters_from_current_pool = parse_clusters(cluster_file)
        
        # Split the current pool's representatives to find *new* test representatives for this level
        _, _, new_test_reps = split_representatives(
            clusters_from_current_pool,
            test_ratio=args.test_ratio,
            val_ratio=0,  # No validation set here
            min_label_count=args.min_label_count_for_split,
            random_state=args.random_state
        )
        
        logging.info(f"Identified {len(new_test_reps)} new representative sequences for the S{identity_percent} test set.")
        
        # Add the new representatives to our cumulative set
        cumulative_test_reps_headers.update(new_test_reps)
        
        # The test set for this level consists of all *representatives* identified so far.
        final_test_sets[identity_percent] = {
            rep: all_sequences[rep] for rep in cumulative_test_reps_headers if rep in all_sequences
        }
        logging.info(f"Cumulative test set for S{identity_percent} now contains {len(final_test_sets[identity_percent])} representative sequences.")

        # Find all members of the new test clusters and remove them from the pool to prevent data leakage.
        members_to_remove = set()
        for rep in new_test_reps:
            members_to_remove.update(clusters_from_current_pool.get(rep, [rep]))
        
        # Shrink the pool for the next iteration.
        current_pool_sequences = {
            h: s for h, s in current_pool_sequences.items() if h not in members_to_remove
        }
        
        logging.info(f"Removed {len(members_to_remove)} sequences (members of new test clusters). Pool size: {len(current_pool_sequences)}.")

        # Clean up the temporary file for this iteration
        if pool_fasta_path.exists():
            pool_fasta_path.unlink()

    # The final training set is what remains in the pool after all test sets have been carved out.
    final_training_set = current_pool_sequences

    # --- Final Label Consistency Check ---
    logging.info("\n--- Step 3: Final Label Consistency Check ---")
    final_train_labels = {get_cath_label(h) for h in final_training_set.keys()}
    logging.info(f"Final training set contains {len(final_train_labels)} unique labels.")

    # Filter validation set
    original_val_len = len(final_validation_set)
    final_validation_set = {h: s for h, s in final_validation_set.items() if get_cath_label(h) in final_train_labels}
    logging.info(f"Filtered validation set from {original_val_len} to {len(final_validation_set)} sequences to match training labels.")

    # Filter all test sets
    for identity in final_test_sets:
        original_test_len = len(final_test_sets[identity])
        final_test_sets[identity] = {
            h: s for h, s in final_test_sets[identity].items() if get_cath_label(h) in final_train_labels
        }
        logging.info(f"Filtered S{identity} test set from {original_test_len} to {len(final_test_sets[identity])} sequences.")

    # --- Save aggregated and filtered datasets ---
    # Save the single, final training set
    unified_dir = output_base_dir / "unified_train_set"
    unified_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(unified_dir / 'train.fasta', final_training_set)
    write_fasta(unified_dir / 'valid.fasta', final_validation_set)
    logging.info(f"\nSaved final training set to {unified_dir / 'train.fasta'}")
    logging.info(f"Saved final validation set to {unified_dir / 'valid.fasta'}")

    # Save each test set in its identity-specific directory
    for identity, test_seqs in final_test_sets.items():
        identity_dir = output_base_dir / f"s{identity}"
        identity_dir.mkdir(parents=True, exist_ok=True)
        write_fasta(identity_dir / 'test.fasta', test_seqs)
        logging.info(f"Saved S{identity} test set to {identity_dir / 'test.fasta'}")

    # --- Update and Print Final Summary Table ---
    summary_text = generate_summary(final_training_set, final_validation_set, final_test_sets)

    # Print to console
    print("\n" + summary_text)

    # Save to file
    if args.summary_file:
        summary_file_path = output_base_dir / args.summary_file
        try:
            with open(summary_file_path, 'w') as f:
                f.write(summary_text)
            logging.info(f"Summary table saved to {summary_file_path}")
        except IOError as e:
            logging.error(f"Failed to write summary file to {summary_file_path}: {e}")


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
    parser.add_argument(
        "--summary_file",
        type=str,
        default="summary.txt",
        help="Name of the file in the output directory to save the final summary table. If empty, not saved."
    )
    
    args = parser.parse_args()
    main(args) 