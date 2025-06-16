"""
Splits a dataset of sequences with CATH superfamily labels into training,
validation, and test sets.

This script performs a stratified split based on CATH superfamily labels to ensure
that each data split contains a representative sample of each superfamily.

Superfamilies with fewer than a specified number of members (default is 3) are
filtered out and stored separately, as they cannot be reliably split into
training, validation, and testing sets.
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Constants ---
MIN_SUPERFAMILY_SIZE = 3
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
RANDOM_SEED = 42


def group_by_superfamily(
    data: Dict[str, str]
) -> Dict[str, List[str]]:
    """Groups sequence IDs by their CATH superfamily label.

    Args:
        data: A dictionary mapping sequence IDs to superfamily labels.

    Returns:
        A dictionary mapping superfamily labels to a list of sequence IDs.
    """
    logging.info("Grouping sequences by superfamily...")
    superfamily_groups = defaultdict(list)
    for seq_id, sf_label in tqdm(data.items(), desc="Grouping by superfamily"):
        superfamily_groups[sf_label].append(seq_id)
    logging.info(f"Found {len(superfamily_groups)} unique superfamilies.")
    return superfamily_groups


def filter_superfamilies(
    superfamily_groups: Dict[str, List[str]], min_size: int
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Filters out superfamilies with fewer than a minimum number of sequences.

    Args:
        superfamily_groups: A dictionary mapping superfamilies to sequence IDs.
        min_size: The minimum number of sequences required for a superfamily
                  to be included in the main splits.

    Returns:
        A tuple containing two dictionaries:
        - The first for superfamilies large enough for splitting.
        - The second for small superfamilies.
    """
    logging.info(
        f"Filtering out superfamilies with fewer than {min_size} members..."
    )
    large_enough_superfamilies = {}
    small_superfamilies = {}
    for sf, sequences in tqdm(
        superfamily_groups.items(), desc="Filtering superfamilies"
    ):
        if len(sequences) < min_size:
            small_superfamilies[sf] = sequences
        else:
            large_enough_superfamilies[sf] = sequences

    logging.info(
        f"Found {len(large_enough_superfamilies)} superfamilies for splitting."
    )
    logging.info(
        f"Found {len(small_superfamilies)} small superfamilies (will be set aside)."
    )
    return large_enough_superfamilies, small_superfamilies


def create_splits(
    superfamily_groups: Dict[str, List[str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Performs a stratified split of sequences into train, validation, and test sets.

    Args:
        superfamily_groups: Superfamilies to be split.
        val_ratio: The proportion of data to allocate to the validation set.
        test_ratio: The proportion of data to allocate to the test set.
        seed: A random seed for reproducibility.

    Returns:
        A tuple of dictionaries for the train, validation, and test splits.
    """
    logging.info("Creating train, validation, and test splits...")
    train_data, val_data, test_data = {}, {}, {}
    rng = random.Random(seed)

    for sf_label, sequences in tqdm(
        superfamily_groups.items(), desc="Creating splits"
    ):
        rng.shuffle(sequences)
        n = len(sequences)

        # Ensure at least one sample in test and validation sets
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        # The remaining sequences go to the training set
        test_seqs = sequences[:n_test]
        val_seqs = sequences[n_test : n_test + n_val]
        train_seqs = sequences[n_test + n_val :]

        for seq_id in train_seqs:
            train_data[seq_id] = sf_label
        for seq_id in val_seqs:
            val_data[seq_id] = sf_label
        for seq_id in test_seqs:
            test_data[seq_id] = sf_label

    logging.info(f"Training set size: {len(train_data)} sequences")
    logging.info(f"Validation set size: {len(val_data)} sequences")
    logging.info(f"Test set size: {len(test_data)} sequences")
    return train_data, val_data, test_data


def flatten_superfamily_groups(groups: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Converts a grouped superfamily dictionary back to a flat sequence map.
    """
    flat_map = {}
    for sf_label, sequences in groups.items():
        for seq_id in sequences:
            flat_map[seq_id] = sf_label
    return flat_map


def save_json(data: Dict, file_path: Path):
    """Saves a dictionary to a JSON file.

    Args:
        data: The dictionary to save.
        file_path: The path to the output JSON file.
    """
    logging.info(f"Saving data to {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """Main function to orchestrate the dataset splitting process."""
    parser = argparse.ArgumentParser(description="Split CATH dataset.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default="data/TED/TED-SF-mapping.json",
        help="Path to the input JSON file with sequence-to-superfamily mappings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/TED/splits",
        help="Directory to save the output split files.",
    )
    parser.add_argument(
        "--min-sf-size",
        type=int,
        default=MIN_SUPERFAMILY_SIZE,
        help="Minimum sequences per superfamily to be included in splits.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Proportion of data for the validation set.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Proportion of data for the test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # --- 1. Load Data ---
    try:
        with open(args.input_file, "r") as f:
            sequence_mappings = json.load(f)
        logging.info(f"Successfully loaded {len(sequence_mappings)} sequences.")
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {args.input_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {args.input_file}")
        return

    # --- 2. Group and Filter ---
    superfamily_groups = group_by_superfamily(sequence_mappings)
    large_sf, small_sf = filter_superfamilies(
        superfamily_groups, args.min_sf_size
    )

    # --- 3. Create Splits ---
    train_data, val_data, test_data = create_splits(
        large_sf, args.val_ratio, args.test_ratio, args.seed
    )

    # --- 4. Prepare small superfamily data for saving ---
    small_superfamilies_data = flatten_superfamily_groups(small_sf)

    # --- 5. Save all files ---
    save_json(train_data, args.output_dir / "s100_train.json")
    save_json(val_data, args.output_dir / "s100_val.json")
    save_json(test_data, args.output_dir / "s100_test.json")
    save_json(
        small_superfamilies_data, args.output_dir / "small_superfamilies.json"
    )

    logging.info("Dataset splitting process completed successfully.")


if __name__ == "__main__":
    main()
