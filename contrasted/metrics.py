import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from distances import pairwise_distance
from utils import get_logger

log = get_logger(__name__)

# Cache for "holdout vs reference" indices per stage ('val' / 'test')
_holdout_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def compute_centroid_metrics(
    embeddings: Tensor,
    labels: Tensor,
    stage: str,
    min_class_size: int = 2,
) -> Tuple[Dict[str, float], Optional[Tensor]]:
    """
    Computes nearest‐centroid classification metrics entirely on CPU.
    Ignores classes with fewer than `min_class_size` samples for centroid calculation
    and evaluates only on samples from eligible classes.
    Returns metrics dict and indices of evaluated samples relative to input.
    """
    metrics: Dict[str, float] = {}
    metric_prefix = f"{stage}/centroid"
    eval_indices: Optional[Tensor] = None # Store evaluated indices

    try:
        with torch.no_grad():
            # move off GPU ASAP
            embs = embeddings.detach().cpu()
            labs = labels.detach().cpu()

            # 1) Identify classes with enough samples
            unique_labels, counts = torch.unique(labs, return_counts=True)
            eligible_classes = unique_labels[counts >= min_class_size]

            if eligible_classes.numel() == 0:
                log.warning(f"Centroid metrics skipped for {stage}: no classes with >= {min_class_size} samples.")
                for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                    metrics[f"{metric_prefix}_{name}"] = 0.0
                return metrics, None # Return None for indices

            # 2) Compute centroids *only* for eligible classes using all their samples
            centroids = torch.stack([embs[labs == c].mean(dim=0) for c in eligible_classes])

            # 3) Filter embeddings/labels to evaluate *only* on samples from eligible classes
            eligible_mask = torch.isin(labs, eligible_classes)
            eval_embs = embs[eligible_mask]
            eval_labs = labs[eligible_mask]
            eval_indices = torch.arange(embs.size(0))[eligible_mask] # Store filtered original indices

            if eval_embs.numel() == 0: # Should not happen if eligible_classes is not empty, but safety check
                log.warning(f"Centroid metrics skipped for {stage}: No eligible samples found after filtering.")
                for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                    metrics[f"{metric_prefix}_{name}"] = 0.0
                return metrics, None # Return None for indices

            # 4) Compute pairwise distances between evaluation embeddings and eligible centroids
            dists = pairwise_distance(eval_embs, centroids) # Shape: [n_eval_samples, n_eligible_classes]

            # 5) Assign nearest eligible centroid
            preds_indices = torch.argmin(dists, dim=1)
            preds = eligible_classes[preds_indices] # Map indices back to original class labels

            # 6) Compute numpy/sklearn metrics on the filtered set
            y_true = eval_labs.numpy()
            y_pred = preds.numpy()
            metrics[f"{metric_prefix}_acc"]          = accuracy_score(y_true, y_pred)
            # Use labels=eligible_classes.numpy() if using average='weighted' or 'micro' might be needed
            metrics[f"{metric_prefix}_balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
            metrics[f"{metric_prefix}_precision"]    = precision_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_recall"]       = recall_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_f1_macro"]     = f1_score(y_true, y_pred, average="macro", zero_division=0)

    except Exception as e:
        log.error(f"Error computing centroid metrics for {stage}: {e}", exc_info=True)
        # on any error, return zeros to keep training stable
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{metric_prefix}_{name}"] = 0.0
        return metrics, None # Return None indices on error

    return metrics, eval_indices # Return calculated metrics and the indices used


def compute_knn_metrics(
    embeddings: Tensor,
    labels: Tensor,
    k: int,
    stage: str,
    knn_batch_size: int = 1024,
    num_workers: int = 0,
    min_class_size: int = 2,
) -> Tuple[Dict[str, float], Optional[Tensor]]:
    """
    Leave‑one‑out k‑NN on the embeddings (CPU), batched for memory efficiency.
    Evaluates metrics only on samples whose true class has at least `min_class_size` members.
    Returns metrics dict and indices of evaluated samples relative to input.
    """
    metrics = {}
    metric_prefix = f"{stage}/knn_{k}"
    eval_indices: Optional[Tensor] = None # Store evaluated indices
    n_samples = embeddings.size(0)
    original_indices = torch.arange(n_samples) # Track original indices

    if n_samples <= k:
        log.warning(f"k-NN computation skipped for k={k}, stage={stage}: Not enough samples ({n_samples})")
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
             metrics[f"{metric_prefix}_{name}"] = 0.0
        return metrics, None

    try:
        with torch.no_grad():
            embs = embeddings.detach().cpu()
            labs = labels.detach().cpu()

            # --- Pre-check: Identify eligible classes ---
            unique_labels, counts = torch.unique(labs, return_counts=True)
            eligible_classes = unique_labels[counts >= min_class_size]

            if eligible_classes.numel() == 0:
                log.warning(f"k-NN ({k}) metrics skipped for {stage}: no classes with >= {min_class_size} samples.")
                for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                    metrics[f"{metric_prefix}_{name}"] = 0.0
                return metrics, None
            # --- End Pre-check ---

            all_preds = []
            # Process in batches to avoid large N x N matrix
            for i in range(0, n_samples, knn_batch_size):
                batch_indices = range(i, min(i + knn_batch_size, n_samples))
                embs_batch = embs[batch_indices]

                # Compute distances between batch and all samples
                dists_batch = pairwise_distance(embs_batch, embs) # Shape: [batch_size, n_samples]

                # Exclude self-distances within the batch vs its position in the full set
                for batch_idx, global_idx in enumerate(batch_indices):
                    if global_idx < dists_batch.shape[1]: # Ensure index is valid
                        dists_batch[batch_idx, global_idx] = float("inf")

                # Find k-nearest neighbors for the current batch
                actual_k = min(k, n_samples - 1)
                if actual_k <= 0: raise ValueError(f"Cannot compute k-NN with k={actual_k}")
                knn_idxs_batch = torch.topk(dists_batch, actual_k, largest=False).indices # Shape: [batch_size, k]

                # Get neighbor labels and majority vote for the batch
                neighbor_labels_batch = labs[knn_idxs_batch] # Shape: [batch_size, k]
                preds_batch, _ = torch.mode(neighbor_labels_batch, dim=1) # Shape: [batch_size]
                all_preds.append(preds_batch)

            # Concatenate predictions from all batches
            preds = torch.cat(all_preds)

            # --- Filter results for metric calculation ---
            eligible_mask = torch.isin(labs, eligible_classes)
            y_true_filtered = labs[eligible_mask]
            y_pred_filtered = preds[eligible_mask]
            eval_indices = original_indices[eligible_mask] # Store filtered original indices

            if len(y_true_filtered) == 0:
                 log.warning(f"k-NN ({k}) metrics skipped for {stage}: no eligible samples remained after filtering.")
                 for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                     metrics[f"{metric_prefix}_{name}"] = 0.0
                 return metrics, None
            # --- End Filter ---

            # Compute sklearn metrics using filtered results
            # Compute sklearn metrics
            y_true = y_true_filtered.numpy()
            y_pred = y_pred_filtered.numpy()
            metrics[f"{metric_prefix}_acc"]           = accuracy_score(y_true, y_pred)
            metrics[f"{metric_prefix}_balanced_acc"]  = balanced_accuracy_score(y_true, y_pred)
            metrics[f"{metric_prefix}_precision"]     = precision_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_recall"]        = recall_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_f1_macro"]      = f1_score(y_true, y_pred, average="macro", zero_division=0)

            # Optional: Keep limited debug prints if helpful
            # print(f"  [knn k={k} batch {i//knn_batch_size}] preds_batch sample:", preds_batch[:3])

    except Exception as e:
        log.error(f"Error in compute_knn_metrics(k={k}, stage={stage}): {e}", exc_info=True)
        # Ensure metrics dictionary exists and provide defaults on error
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{metric_prefix}_{name}"] = 0.0
        # Optionally re-raise if you want training to stop on metric calculation errors
        # raise e
        return metrics, None # Return None indices on error

    return metrics, eval_indices # Return calculated metrics and the indices used


def compute_centroid_metrics_reference(
    test_embeddings: Tensor,
    test_labels: Tensor,
    ref_embeddings: Tensor,
    ref_labels: Tensor,
    stage: str
) -> Dict[str, float]:
    """Classify test points by nearest‐centroid computed from the reference (training) set."""
    metrics: Dict[str, float] = {}
    try:
        classes = torch.unique(ref_labels)
        centroids = torch.stack([ref_embeddings[ref_labels == c].mean(dim=0) for c in classes])
        dists = pairwise_distance(test_embeddings, centroids)
        preds = classes[torch.argmin(dists, dim=1)]
        y_true = test_labels.numpy()
        y_pred = preds.numpy()
        metrics[f"{stage}/centroid_acc"]          = accuracy_score(y_true, y_pred)
        metrics[f"{stage}/centroid_balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
        metrics[f"{stage}/centroid_precision"]    = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics[f"{stage}/centroid_recall"]       = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics[f"{stage}/centroid_f1_macro"]     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception:
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{stage}/centroid_{name}"] = 0.0
    return metrics


def compute_knn_metrics_reference(
    test_embeddings: Tensor,
    test_labels: Tensor,
    ref_embeddings: Tensor,
    ref_labels: Tensor,
    k: int,
    stage: str,
    knn_batch_size: int = 1024,
) -> Dict[str, float]:
    """k‑NN on test points against reference (training) embeddings."""
    metrics: Dict[str, float] = {}
    num_ref = ref_embeddings.size(0)
    if num_ref <= k:
        log.warning(f"k-NN ref skipped for k={k}, stage={stage}: only {num_ref} ref samples")
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{stage}/knn_{k}_{name}"] = 0.0
        return metrics

    try:
        all_preds = []
        for i in range(0, test_embeddings.size(0), knn_batch_size):
            batch_slice = slice(i, min(i + knn_batch_size, test_embeddings.size(0)))
            test_batch = test_embeddings[batch_slice]
            dists = pairwise_distance(test_batch, ref_embeddings)
            knn_idxs = torch.topk(dists, k, largest=False).indices
            neighbor_labels = ref_labels[knn_idxs]
            preds_batch, _ = torch.mode(neighbor_labels, dim=1)
            all_preds.append(preds_batch)
        preds = torch.cat(all_preds)
        y_true = test_labels.numpy()
        y_pred = preds.numpy()
        metrics[f"{stage}/knn_{k}_acc"]           = accuracy_score(y_true, y_pred)
        metrics[f"{stage}/knn_{k}_balanced_acc"]  = balanced_accuracy_score(y_true, y_pred)
        metrics[f"{stage}/knn_{k}_precision"]     = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics[f"{stage}/knn_{k}_recall"]        = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics[f"{stage}/knn_{k}_f1_macro"]      = f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception as e:
        log.error(f"Error in compute_knn_metrics_reference(k={k}, stage={stage}): {e}", exc_info=True)
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{stage}/knn_{k}_{name}"] = 0.0
    return metrics


# -------------------------------------------------------------------
# MODIFIED: hold‐out CENTROID evaluation (cached per split)
# -------------------------------------------------------------------
def compute_holdout_metrics(
    embeddings: Tensor,
    labels: Tensor,
    stage: str,
    holdout_size: int = 300,
    min_class_size_for_holdout: int = 2,
    seed: int = 42
) -> Dict[str, float]:
    """
    Hold‐out evaluation using only NEAREST CENTROID.
    Selects `holdout_size` samples once (from classes with ≥ min_class_size_for_holdout)
    and caches those indices for reuse across the entire run.
    """
    metrics: Dict[str, float] = {}
    metric_prefix = f"{stage}/holdout_centroid"

    try:
        with torch.no_grad():
            # move data to CPU
            embs = embeddings.detach().cpu()
            labs = labels.detach().cpu()
            n = embs.size(0)

            # 1) find labels with enough samples
            unique_labels, counts = torch.unique(labs, return_counts=True)
            eligible = unique_labels[counts >= min_class_size_for_holdout]
            if eligible.numel() == 0:
                log.warning(f"Holdout skipped for {stage}: no classes ≥ {min_class_size_for_holdout}")
                for m in ("acc","balanced_acc","precision","recall","f1_macro"):
                    metrics[f"{metric_prefix}_{m}"] = 0.0
                return metrics

            idxs = torch.where(torch.isin(labs, eligible))[0]
            if idxs.numel() < holdout_size:
                log.warning(f"Holdout skipped for {stage}: only {idxs.numel()} eligible (<{holdout_size})")
                for m in ("acc","balanced_acc","precision","recall","f1_macro"):
                    metrics[f"{metric_prefix}_{m}"] = 0.0
                return metrics

            # 2) reuse or compute+cache the holdout/ref split
            if stage in _holdout_cache:
                hold_idx, ref_idx = _holdout_cache[stage]
            else:
                gen = torch.Generator().manual_seed(seed)
                perm = torch.randperm(idxs.numel(), generator=gen)
                hold_idx = idxs[perm[:holdout_size]]
                mask = torch.ones(n, dtype=torch.bool)
                mask[hold_idx] = False
                ref_idx = torch.where(mask)[0]
                _holdout_cache[stage] = (hold_idx, ref_idx)

            # 3) partition embeddings / labels
            hold_embs, hold_labs = embs[hold_idx], labs[hold_idx]
            ref_embs, ref_labs   = embs[ref_idx], labs[ref_idx]

            # 4) compute centroids from REF and assign HOLD‐OUT
            classes = torch.unique(ref_labs)
            centroids = torch.stack([ref_embs[ref_labs == c].mean(dim=0) for c in classes])
            dists = pairwise_distance(hold_embs, centroids)
            preds = classes[torch.argmin(dists, dim=1)]

            # 5) sklearn metrics
            y_true = hold_labs.numpy()
            y_pred = preds.numpy()
            metrics[f"{metric_prefix}_acc"]          = accuracy_score(y_true, y_pred)
            metrics[f"{metric_prefix}_balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
            metrics[f"{metric_prefix}_precision"]    = precision_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_recall"]       = recall_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{metric_prefix}_f1_macro"]     = f1_score(y_true, y_pred, average="macro", zero_division=0)

    except Exception as e:
        log.error(f"Error in compute_holdout_metrics for {stage}: {e}", exc_info=True)
        # on error/fallback
        for m in ("acc","balanced_acc","precision","recall","f1_macro"):
            metrics[f"{metric_prefix}_{m}"] = 0.0

    return metrics