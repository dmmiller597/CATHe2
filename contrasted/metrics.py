import torch
from torch import Tensor
from typing import Dict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from distances import pairwise_distance
from utils import get_logger

log = get_logger(__name__)


def compute_centroid_metrics(
    embeddings: Tensor,
    labels: Tensor,
    stage: str,
) -> Dict[str, float]:
    """Computes nearest‐centroid classification metrics entirely on CPU."""
    metrics: Dict[str, float] = {}
    try:
        with torch.no_grad():
            # move off GPU ASAP
            embs = embeddings.detach().cpu()
            labs = labels.detach().cpu()

            # 1) compute one centroid per class
            classes = torch.unique(labs)
            centroids = torch.stack([embs[labs == c].mean(dim=0) for c in classes])

            # 2) pairwise squared-euclidean distances (CPU)
            dists = pairwise_distance(embs, centroids)

            # 3) assign nearest centroid
            preds = classes[torch.argmin(dists, dim=1)]

            # 4) compute numpy/sklearn metrics
            y_true = labs.numpy()
            y_pred = preds.numpy()
            metrics[f"{stage}/centroid_acc"]          = accuracy_score(y_true, y_pred)
            metrics[f"{stage}/centroid_balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
            metrics[f"{stage}/centroid_precision"]    = precision_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{stage}/centroid_recall"]       = recall_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{stage}/centroid_f1_macro"]     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception:
        # on any error, return zeros to keep training stable
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{stage}/centroid_{name}"] = 0.0
    return metrics


def compute_knn_metrics(
    embeddings: Tensor,
    labels: Tensor,
    k: int,
    stage: str,
    knn_batch_size: int = 1024,
    num_workers: int = 0
) -> Dict[str, float]:
    """Leave‑one‑out k‑NN on the embeddings (CPU), batched for memory efficiency."""
    metrics = {}
    n_samples = embeddings.size(0)
    if n_samples <= k:
        log.warning(f"k-NN computation skipped for k={k}, stage={stage}: Not enough samples ({n_samples})")
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
             metrics[f"{stage}/knn_{k}_{name}"] = 0.0
        return metrics

    try:
        with torch.no_grad():
            embs = embeddings.detach().cpu()
            labs = labels.detach().cpu()
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
                knn_idxs_batch = torch.topk(dists_batch, k, largest=False).indices # Shape: [batch_size, k]

                # Get neighbor labels and majority vote for the batch
                neighbor_labels_batch = labs[knn_idxs_batch] # Shape: [batch_size, k]
                preds_batch, _ = torch.mode(neighbor_labels_batch, dim=1) # Shape: [batch_size]
                all_preds.append(preds_batch)

            # Concatenate predictions from all batches
            preds = torch.cat(all_preds)

            # Compute sklearn metrics
            y_true = labs.numpy()
            y_pred = preds.numpy()
            metrics[f"{stage}/knn_{k}_acc"]           = accuracy_score(y_true, y_pred)
            metrics[f"{stage}/knn_{k}_balanced_acc"]  = balanced_accuracy_score(y_true, y_pred)
            metrics[f"{stage}/knn_{k}_precision"]     = precision_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{stage}/knn_{k}_recall"]        = recall_score(y_true, y_pred, average="macro", zero_division=0)
            metrics[f"{stage}/knn_{k}_f1_macro"]      = f1_score(y_true, y_pred, average="macro", zero_division=0)

            # Optional: Keep limited debug prints if helpful
            # print(f"  [knn k={k} batch {i//knn_batch_size}] preds_batch sample:", preds_batch[:3])

    except Exception as e:
        log.error(f"Error in compute_knn_metrics(k={k}, stage={stage}): {e}", exc_info=True)
        # Ensure metrics dictionary exists and provide defaults on error
        for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
            metrics[f"{stage}/knn_{k}_{name}"] = 0.0
        # Optionally re-raise if you want training to stop on metric calculation errors
        # raise e
    return metrics


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