# contrasted/evaluate_baseline_embeddings.py
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml
import csv

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

# Import utilities directly from the contrasted package
from .data import CATH_LEVEL_NAMES, get_level_label # Use relative import
from .distances import pairwise_distance # Use relative import

# Core Evaluation Logic (adapted from contrasted/metrics.py)

def evaluate_split(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    min_class_size: int,
    split_name: str,
    knn_batch_size: int = 1024,
) -> Dict[str, float]:
    """
    Evaluates embeddings using nearest centroid and 1-NN (leave-one-out)
    after filtering for minimum class size.
    """
    print(f"\n--- Evaluating {split_name} split ---")
    metrics = {}
    n_samples = embeddings.size(0)

    if n_samples == 0:
        print("No samples found.")
        return {}

    # 1. Identify classes with enough samples for reliable evaluation
    unique_labels, counts = torch.unique(labels, return_counts=True)
    eligible_classes = unique_labels[counts >= min_class_size]

    if eligible_classes.numel() == 0:
        print(f"Skipping evaluation: No classes found with >= {min_class_size} samples.")
        # Return zero metrics for consistency
        for metric_type in ["centroid", "knn_1"]:
             for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                 metrics[f"{split_name}/{metric_type}_{name}"] = 0.0
        return metrics

    print(f"Found {eligible_classes.numel()} classes with >= {min_class_size} samples.")

    # 2. Filter embeddings/labels to evaluate *only* on samples from eligible classes
    eligible_mask = torch.isin(labels, eligible_classes)
    eval_embs = embeddings[eligible_mask]
    eval_labs = labels[eligible_mask]
    n_eval_samples = eval_embs.size(0)

    if n_eval_samples == 0:
        print(f"Skipping evaluation: No samples remained after filtering for eligible classes.")
        # Return zero metrics for consistency
        for metric_type in ["centroid", "knn_1"]:
             for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                 metrics[f"{split_name}/{metric_type}_{name}"] = 0.0
        return metrics

    print(f"Evaluating on {n_eval_samples} samples from eligible classes.")
    y_true_np = eval_labs.numpy() # Common true labels for both methods

    # --- 3. Centroid Evaluation ---
    print("Calculating Centroid metrics...")
    centroid_start = time.time()
    metric_prefix = f"{split_name}/centroid"
    try:
        # Compute centroids *only* for eligible classes using all eligible samples
        # Note: This uses the *evaluation set* to compute centroids, matching the original logic.
        centroids = torch.stack([eval_embs[eval_labs == c].mean(dim=0) for c in eligible_classes])

        # Compute distances between evaluation embeddings and eligible centroids
        dists = pairwise_distance(eval_embs, centroids) # Shape: [n_eval_samples, n_eligible_classes]

        # Assign nearest eligible centroid
        preds_indices = torch.argmin(dists, dim=1)
        preds = eligible_classes[preds_indices] # Map indices back to original class labels
        y_pred_np = preds.numpy()

        # Compute numpy/sklearn metrics on the filtered set
        metrics[f"{metric_prefix}_acc"]          = accuracy_score(y_true_np, y_pred_np)
        metrics[f"{metric_prefix}_balanced_acc"] = balanced_accuracy_score(y_true_np, y_pred_np)
        metrics[f"{metric_prefix}_precision"]    = precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        metrics[f"{metric_prefix}_recall"]       = recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        metrics[f"{metric_prefix}_f1_macro"]     = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        print(f"Centroid metrics calculated in {time.time() - centroid_start:.2f}s")

    except Exception as e:
        print(f"Error computing centroid metrics for {split_name}: {e}")
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{metric_prefix}_{name}"] = 0.0


    # --- 4. k-NN (k=1) Evaluation (Leave-One-Out within eligible set) ---
    print("Calculating k-NN (k=1) LOO metrics...")
    knn_start = time.time()
    metric_prefix = f"{split_name}/knn_1"
    k = 1
    if n_eval_samples <= k:
         print(f"Skipping k-NN (k=1): Not enough eligible samples ({n_eval_samples})")
         for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{metric_prefix}_{name}"] = 0.0
    else:
        try:
            all_preds_knn = []
            # Process in batches for distance calculation
            for i in range(0, n_eval_samples, knn_batch_size):
                batch_indices = range(i, min(i + knn_batch_size, n_eval_samples))
                embs_batch = eval_embs[batch_indices]

                # Compute distances between batch and *all eligible* samples
                dists_batch = pairwise_distance(embs_batch, eval_embs) # Shape: [batch_size, n_eval_samples]

                # Exclude self-distances (important for leave-one-out)
                eligible_original_indices = torch.where(eligible_mask)[0] # Indices in the *full* dataset

                for batch_idx, eval_idx in enumerate(batch_indices):
                     # The column corresponding to self in the eval dist matrix is simply eval_idx
                    self_dist_column_idx = eval_idx
                    if self_dist_column_idx < dists_batch.shape[1]:
                         dists_batch[batch_idx, self_dist_column_idx] = float("inf")

                # Find 1 nearest neighbor (excluding self) for the current batch
                knn_idxs_batch = torch.topk(dists_batch, k, largest=False).indices # Shape: [batch_size, k]

                # Get neighbor labels and majority vote (trivial for k=1)
                neighbor_labels_batch = eval_labs[knn_idxs_batch] # Shape: [batch_size, k]
                preds_batch = neighbor_labels_batch.squeeze(dim=1) # Shape: [batch_size] for k=1
                all_preds_knn.append(preds_batch)

            # Concatenate predictions from all batches
            preds_knn = torch.cat(all_preds_knn)
            y_pred_knn_np = preds_knn.numpy()

            # Compute sklearn metrics using filtered results
            metrics[f"{metric_prefix}_acc"]           = accuracy_score(y_true_np, y_pred_knn_np)
            metrics[f"{metric_prefix}_balanced_acc"]  = balanced_accuracy_score(y_true_np, y_pred_knn_np)
            metrics[f"{metric_prefix}_precision"]     = precision_score(y_true_np, y_pred_knn_np, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_recall"]        = recall_score(y_true_np, y_pred_knn_np, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_f1_macro"]      = f1_score(y_true_np, y_pred_knn_np, average="macro", zero_division=0)
            print(f"k-NN (k=1) LOO metrics calculated in {time.time() - knn_start:.2f}s")

        except Exception as e:
            print(f"Error computing k-NN (k=1) metrics for {split_name}: {e}")
            for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                 metrics[f"{metric_prefix}_{name}"] = 0.0

    return metrics

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
             print(f"Error: 'data_dir' not found in config file {args.config_file}")
             return
        base_dir = Path(data_config['data_dir']).resolve()
        print(f"Using data directory: {base_dir}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_file}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {args.config_file}: {e}")
        return
    except Exception as e:
         print(f"An unexpected error occurred loading the config: {e}")
         return


    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        emb_key = f"{split}_embeddings"
        lbl_key = f"{split}_labels"

        # Check if paths exist in config
        if emb_key not in data_config or lbl_key not in data_config:
             print(f"Skipping {split} split - '{emb_key}' or '{lbl_key}' not found in {args.config_file}.")
             continue

        emb_file_rel = data_config[emb_key]
        lbl_file_rel = data_config[lbl_key]

        if not emb_file_rel or not lbl_file_rel:
            print(f"Skipping {split} split - missing file paths in config for this split.")
            continue

        emb_path = base_dir / emb_file_rel
        lbl_path = base_dir / lbl_file_rel

        if not emb_path.is_file():
            print(f"ERROR: Embeddings file not found: {emb_path}")
            continue
        if not lbl_path.is_file():
            print(f"ERROR: Labels file not found: {lbl_path}")
            continue

        print(f"\nLoading data for {split} split...")
        try:
            # Load embeddings
            emb_data = np.load(emb_path)
            if "embeddings" not in emb_data:
                print(f"ERROR: 'embeddings' key not found in {emb_path}")
                continue
            embeddings_np = emb_data["embeddings"].astype(np.float32)
            embeddings_tensor = torch.from_numpy(embeddings_np)

            # Load labels
            df = pd.read_csv(lbl_path)
            if "SF" not in df.columns:
                print(f"ERROR: 'SF' column not found in {lbl_path}")
                continue

            # Process labels using imported function
            sf_series = df["SF"].astype(str).apply(lambda s: get_level_label(s, args.cath_level))
            encoder = LabelEncoder()
            labels_encoded = encoder.fit_transform(sf_series)
            labels_tensor = torch.from_numpy(labels_encoded).long()

            # Check length consistency
            if len(embeddings_tensor) != len(labels_tensor):
                 print(f"ERROR: Mismatch lengths for {split}: embeddings={len(embeddings_tensor)}, labels={len(labels_tensor)}")
                 continue

            print(f"Loaded {len(embeddings_tensor)} samples.")

            # Evaluate
            split_results = evaluate_split(
                embeddings=embeddings_tensor,
                labels=labels_tensor,
                min_class_size=args.min_class_size,
                split_name=split,
                knn_batch_size=args.knn_batch_size,
            )
            results.update(split_results)

        except Exception as e:
            print(f"Failed to process {split} split: {e}")

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