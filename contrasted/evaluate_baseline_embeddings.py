# contrasted/evaluate_baseline_embeddings.py
import argparse
import time
from pathlib import Path
from typing import Dict
import yaml
import csv

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Import utilities directly from the contrasted package
from data import get_level_label  # Use relative import
from metrics import compute_centroid_metrics, compute_knn_metrics # Use relative import
from utils import get_logger

log = get_logger(__name__)

# The local evaluate_split function has been removed.
# All metric calculations will now use the imported functions from metrics.py.

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline embeddings using Centroid and k-NN (k=1) LOO.")
    parser.add_argument("--config_file", type=str, default="config/data/protT5.yaml", help="Path to the data configuration YAML file.")
    parser.add_argument("--cath_level", type=int, default=3, choices=[0, 1, 2, 3], help="CATH level for labels (0=C, 1=A, 2=T, 3=H).")
    parser.add_argument("--min_class_size", type=int, default=2, help="Minimum samples per class for evaluation.")
    parser.add_argument("--knn_batch_size", type=int, default=1024, help="Batch size for k-NN distance calculations.")
    parser.add_argument("--output_csv", type=str, default="results/baseline_protT5_embeddings_evaluation.csv", help="Optional path to save results to a CSV file.")

    args = parser.parse_args()

    # Load config from YAML
    try:
        with open(args.config_file, 'r') as f:
            data_config = yaml.safe_load(f)
        if not data_config or 'data_dir' not in data_config:
             log.error(f"Error: 'data_dir' not found in config file {args.config_file}")
             return
        base_dir = Path(data_config['data_dir']).resolve()
        log.info(f"Using data directory: {base_dir}")
    except FileNotFoundError:
        log.error(f"Error: Config file not found at {args.config_file}")
        return
    except yaml.YAMLError as e:
        log.error(f"Error parsing YAML file {args.config_file}: {e}")
        return
    except Exception as e:
         log.error(f"An unexpected error occurred loading the config: {e}")
         return

    # --- MODIFIED: Process both 'val' and 'test' splits ---
    splits = ["val", "test"]
    # --- END MODIFICATION ---

    results = {}

    for split in splits:
        emb_key = f"{split}_embeddings"
        lbl_key = f"{split}_labels"

        # Check if paths exist in config
        if emb_key not in data_config or lbl_key not in data_config:
             log.warning(f"Skipping {split} split - '{emb_key}' or '{lbl_key}' not found in {args.config_file}.")
             continue

        emb_file_rel = data_config[emb_key]
        lbl_file_rel = data_config[lbl_key]

        if not emb_file_rel or not lbl_file_rel:
            log.warning(f"Skipping {split} split - missing file paths in config for this split.")
            continue

        emb_path = base_dir / emb_file_rel
        lbl_path = base_dir / lbl_file_rel

        if not emb_path.is_file():
            log.error(f"ERROR: Embeddings file not found: {emb_path}")
            continue
        if not lbl_path.is_file():
            log.error(f"ERROR: Labels file not found: {lbl_path}")
            continue

        log.info(f"Loading data for {split} split...")
        try:
            # Load embeddings
            emb_data = np.load(emb_path)
            if "embeddings" not in emb_data:
                log.error(f"ERROR: 'embeddings' key not found in {emb_path}")
                continue
            embeddings_np = emb_data["embeddings"].astype(np.float32)
            embeddings_tensor = torch.from_numpy(embeddings_np)

            # Load labels
            df = pd.read_csv(lbl_path)
            if "SF" not in df.columns:
                log.error(f"ERROR: 'SF' column not found in {lbl_path}")
                continue

            # Process labels using imported function
            sf_series = df["SF"].astype(str).apply(lambda s: get_level_label(s, args.cath_level))
            encoder = LabelEncoder()
            labels_encoded = encoder.fit_transform(sf_series)
            labels_tensor = torch.from_numpy(labels_encoded).long()

            # Check length consistency
            if len(embeddings_tensor) != len(labels_tensor):
                 log.error(f"ERROR: Mismatch lengths for {split}: embeddings={len(embeddings_tensor)}, labels={len(labels_tensor)}")
                 continue

            log.info(f"Loaded {len(embeddings_tensor)} samples.")

            # --- MODIFIED: Evaluate using functions from metrics.py ---
            log.info(f"--- Evaluating {split} split (min_class_size={args.min_class_size}) ---")

            # 1. Compute Centroid metrics
            centroid_start = time.time()
            log.info("Calculating Centroid metrics...")
            centroid_metrics, _ = compute_centroid_metrics(
                embeddings=embeddings_tensor,
                labels=labels_tensor,
                stage=split,
                min_class_size=args.min_class_size,
            )
            results.update(centroid_metrics)
            log.info(f"Centroid metrics calculated in {time.time() - centroid_start:.2f}s")

            # 2. Compute k-NN (k=1) LOO metrics
            knn_start = time.time()
            log.info("Calculating k-NN (k=1) LOO metrics...")
            knn_metrics, _ = compute_knn_metrics(
                embeddings=embeddings_tensor,
                labels=labels_tensor,
                k=1,
                stage=split,
                knn_batch_size=args.knn_batch_size,
                min_class_size=args.min_class_size,
            )
            results.update(knn_metrics)
            log.info(f"k-NN (k=1) LOO metrics calculated in {time.time() - knn_start:.2f}s")
            # --- END MODIFICATION ---

        except Exception as e:
            log.error(f"Failed to process {split} split: {e}", exc_info=True)

    print("\n--- Final Results (Console) ---")
    if not results:
        print("No results generated.")
    else:
        # Sort results for consistent output order
        sorted_results = dict(sorted(results.items()))
        for key, value in sorted_results.items():
            print(f"{key}: {value:.4f}")

        # --- Save results to CSV if path provided ---
        if args.output_csv:
            output_path = Path(args.output_csv)
            try:
                # Create parent directory if it doesn't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value']) # Write header
                    for key, value in sorted_results.items():
                        writer.writerow([key, f"{value:.4f}"]) # Write data row
                print(f"\nResults successfully saved to: {output_path}")
            except Exception as e:
                print(f"\nError saving results to CSV ({output_path}): {e}")
        else:
             print("\nResults not saved to CSV (no --output_csv path provided).")
        # --- End save block ---


if __name__ == "__main__":
    main()