import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from utils import get_logger
from distances import pairwise_distance
from losses import SupConLoss
from plotting import generate_tsne_plot, generate_umap_plot

# Module-level logger
log = get_logger(__name__)


def build_projection_network(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float,
    use_layer_norm: bool,
) -> nn.Sequential:
    """Builds an MLP projection head."""
    layers: List[nn.Module] = []
    current = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(current, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        current = h
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def init_weights(module: nn.Module) -> None:
    """Applies Kaiming Normal initialization for Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def compute_intra_inter_distances(
    embeddings: Tensor,
    labels: Tensor,
    max_per_class: int = 1000,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Computes lists of intra-class and inter-class pairwise distances."""
    device = embeddings.device
    intra_list: List[Tensor] = []
    inter_list: List[Tensor] = []
    unique = torch.unique(labels)
    for lbl in unique:
        same = labels == lbl
        diff = ~same
        emb_same = embeddings[same]
        emb_diff = embeddings[diff]
        if emb_same.size(0) > max_per_class:
            idx = torch.randperm(emb_same.size(0), device=device)[:max_per_class]
            emb_same = emb_same[idx]
        if emb_same.size(0) > 1:
            d_same = pairwise_distance(emb_same, emb_same)
            eye = torch.eye(d_same.size(0), dtype=torch.bool, device=device)
            intra_list.append(d_same[~eye])
        if emb_diff.size(0) > max_per_class:
            idx = torch.randperm(emb_diff.size(0), device=device)[:max_per_class]
            emb_diff = emb_diff[idx]
        if emb_same.size(0) > 0 and emb_diff.size(0) > 0:
            d_diff = pairwise_distance(emb_same, emb_diff)
            inter_list.append(d_diff.flatten())
    return intra_list, inter_list


def compute_uniformity(emb_subset: Tensor) -> float:
    """Computes embedding uniformity metric."""
    try:
        pd_sq = torch.pdist(emb_subset).pow(2)
        return pd_sq.mul(-2).exp().mean().add(1e-8).log().item()
    except RuntimeError:
        # fallback to CPU
        cpu = emb_subset.cpu()
        pd_sq = torch.pdist(cpu).pow(2)
        return pd_sq.mul(-2).exp().mean().add(1e-8).log().item()
    except Exception:
        return float("nan")


def compute_triplet_violation(
    intra: Tensor,
    inter: Tensor,
    sample_size: int = 10000,
) -> float:
    """Estimates triplet violation rate."""
    if intra.numel() == 0 or inter.numel() == 0:
        return float("nan")
    dev = intra.device
    n = min(sample_size, intra.numel(), inter.numel())
    i_s = intra[torch.randperm(intra.numel(), device=dev)[:n]]
    e_s = inter[torch.randperm(inter.numel(), device=dev)[:n]]
    return torch.mean((i_s.unsqueeze(1) > e_s.unsqueeze(0)).float()).item()


def compute_distance_metrics(
    embeddings: Tensor,
    labels: Tensor,
    stage: str,
) -> Dict[str, float]:
    """Aggregates distance-based embedding metrics."""
    metrics: Dict[str, float] = {}
    intra_list, inter_list = compute_intra_inter_distances(embeddings, labels)
    if intra_list and inter_list:
        intra_all = torch.cat(intra_list)
        inter_all = torch.cat(inter_list)
        metrics[f"{stage}/mean_intra_dist"] = intra_all.mean().item()
        metrics[f"{stage}/mean_inter_dist"] = inter_all.mean().item()
        metrics[f"{stage}/min_inter_dist"] = inter_all.min().item()
        metrics[f"{stage}/max_intra_dist"] = intra_all.max().item()
        if metrics[f"{stage}/mean_intra_dist"] > 0:
            metrics[f"{stage}/inter_intra_ratio"] = (
                metrics[f"{stage}/mean_inter_dist"]
                / metrics[f"{stage}/mean_intra_dist"]
            )
        metrics[f"{stage}/dist_margin"] = (
            metrics[f"{stage}/mean_inter_dist"] - metrics[f"{stage}/mean_intra_dist"]
        )
        metrics[f"{stage}/class_overlap"] = float(
            torch.mean((intra_all > metrics[f"{stage}/min_inter_dist"]).float())
        )
        # uniformity
        if embeddings.size(0) > 100:
            idx = torch.randperm(embeddings.size(0), device=embeddings.device)[:1000]
            metrics[f"{stage}/embedding_uniformity"] = compute_uniformity(
                embeddings[idx]
            )
        else:
            metrics[f"{stage}/embedding_uniformity"] = float("nan")
        # triplet violation
        metrics[f"{stage}/triplet_violation_rate"] = compute_triplet_violation(
            intra_all, inter_all
        )
    else:
        for key in ["mean_intra_dist", "mean_inter_dist", "embedding_uniformity", "triplet_violation_rate"]:
            metrics[f"{stage}/{key}"] = float("nan")
    return metrics


def compute_knn_streaming(
    embeddings: Tensor,
    labels: Tensor,
    k: int,
    stage: str,
) -> Dict[str, float]:
    """Compute k-NN metrics in a memory-efficient streamed manner."""
    # move everything to CPU
    embs_cpu = embeddings.cpu().float()
    labs_cpu = labels.cpu()
    N = embs_cpu.size(0)
    preds = torch.empty(N, dtype=labs_cpu.dtype)
    # choose a safe chunk size for 48GB A40; fallback to 1024
    chunk_size = 1024  
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        block = embs_cpu[start:end]  # (chunk, dim)
        # compute distances block → all points
        d = torch.cdist(block, embs_cpu)
        # mask self-distances
        idxs = torch.arange(end - start)
        d[idxs, start + idxs] = float('inf')
        if k == 1:
            nearest = torch.argmin(d, dim=1)
            preds[start:end] = labs_cpu[nearest]
        else:
            topk = d.topk(k, dim=1, largest=False)
            neigh = labs_cpu[topk.indices]
            preds[start:end] = torch.mode(neigh, dim=1).values
    y_true = labs_cpu.numpy()
    y_pred = preds.numpy()
    return {
        f"{stage}/knn_acc": accuracy_score(y_true, y_pred),
        f"{stage}/knn_balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def compute_centroid_metrics(
    embeddings: Tensor,
    labels: Tensor,
    stage: str,
) -> Dict[str, float]:
    """Computes nearest-centroid classification accuracy metrics."""
    metrics: Dict[str, float] = {}
    try:
        # compute unique class centroids
        classes = torch.unique(labels)
        centroids = torch.stack([embeddings[labels == c].mean(dim=0) for c in classes])
        # compute distances from embeddings to centroids
        dist_to_centroids = pairwise_distance(embeddings, centroids)
        # assign each embedding to nearest centroid
        idx = torch.argmin(dist_to_centroids, dim=1)
        pred = classes[idx]
        # compute metrics
        y_true = labels.cpu().numpy()
        y_pred = pred.cpu().numpy()
        metrics[f"{stage}/centroid_acc"] = accuracy_score(y_true, y_pred)
        metrics[f"{stage}/centroid_balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
        # compute additional classification metrics
        metrics[f"{stage}/centroid_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics[f"{stage}/centroid_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics[f"{stage}/centroid_f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception:
        metrics[f"{stage}/centroid_acc"] = 0.0
        metrics[f"{stage}/centroid_balanced_acc"] = 0.0
    return metrics


class ContrastiveCATHeModel(L.LightningModule):
    """Lightning module for CATH superfamily contrastive learning."""

    def __init__(
        self,
        input_embedding_dim: int,
        projection_hidden_dims: List[int] = [1024],
        output_embedding_dim: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        use_layer_norm: bool = True,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        knn_val_neighbors: int = 1,
        val_max_samples: int = 100000,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        visualization_method: str = "tsne",
        tsne_viz_dir: str = "results/tsne_plots",
        umap_viz_dir: str = "results/umap_plots",
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Validations
        if not (0 < self.hparams.warmup_start_factor <= 1.0):
            raise ValueError("warmup_start_factor must be > 0 and <= 1.0")
        if self.hparams.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if self.hparams.visualization_method not in ("umap", "tsne"):
            raise ValueError("visualization_method must be 'umap' or 'tsne'.")

        # Model components
        self.projection = build_projection_network(
            input_dim=self.hparams.input_embedding_dim,
            hidden_dims=self.hparams.projection_hidden_dims,
            output_dim=self.hparams.output_embedding_dim,
            dropout=self.hparams.dropout,
            use_layer_norm=self.hparams.use_layer_norm,
        )
        init_weights(self)

        # Supervised contrastive loss
        self.loss_fn = SupConLoss(temperature=self.hparams.temperature)

        # Buffers for metrics
        self._val_outputs: List[Dict[str, Tensor]] = []
        self._test_outputs: List[Dict[str, Tensor]] = []

        # Ensure viz dirs exist
        os.makedirs(self.hparams.tsne_viz_dir, exist_ok=True)
        os.makedirs(self.hparams.umap_viz_dir, exist_ok=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        emb, labels = batch
        proj = self(emb)
        loss = self.loss_fn(proj, labels)
        self.log(
            'train/loss', loss,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        emb, labels = batch
        with torch.inference_mode():
            proj = self(emb)
            # compute validation loss
            loss = self.loss_fn(proj, labels)
        # log validation loss
        self.log(
            'val/loss', loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self._val_outputs.append({"embeddings": proj.detach(), "labels": labels.detach()})

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        emb, labels = batch
        with torch.inference_mode():
            proj = self(emb)
        self._test_outputs.append({"embeddings": proj.cpu().detach(), "labels": labels.cpu().detach()})

    def on_validation_epoch_end(self) -> None:
        m = self._shared_epoch_end(self._val_outputs, 'val')
        if m:
            # filter out NaN metrics
            loggable = {k: v for k, v in m.items() if not (isinstance(v, float) and np.isnan(v))}
            self.log_dict(loggable, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        m = self._shared_epoch_end(self._test_outputs, 'test')
        if m:
            # filter out NaN metrics
            loggable = {k: v for k, v in m.items() if not (isinstance(v, float) and np.isnan(v))}
            self.log_dict(loggable, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # OneCycleLR scheduler for warm-up and annealing per batch
        train_loader = self.trainer.datamodule.train_dataloader()
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.hparams.warmup_epochs / float(self.trainer.max_epochs),
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _shared_epoch_end(self, outputs: List[Dict[str, Tensor]], stage: str) -> Dict[str, float]:
        """Common logic for validation and test epoch ends with error handling."""
        metrics: Dict[str, float] = {}
        if not outputs:
            log.warning(f"No outputs for stage={stage}.")
            # default metrics
            metrics[f"{stage}/knn_acc"] = 0.0
            metrics[f"{stage}/knn_balanced_acc"] = 0.0
            metrics[f"{stage}/centroid_acc"] = 0.0
            metrics[f"{stage}/centroid_balanced_acc"] = 0.0
            outputs.clear()
            return metrics
        try:
            device = self.device
            embs = torch.cat([o['embeddings'].to(device) for o in outputs])
            labs = torch.cat([o['labels'].to(device) for o in outputs])
            # optional visualization
            if (
                stage == 'val'
                and hasattr(self, 'trainer')
                and not self.trainer.sanity_checking
                and self.current_epoch > 0
                and self.current_epoch % 10 == 0
            ):
                log.info(f"Visualizing embeddings (method={self.hparams.visualization_method}, epoch={self.current_epoch}, n={embs.size(0)})")
                if self.hparams.visualization_method == 'umap':
                    generate_umap_plot(self, embs, labs)
                else:
                    generate_tsne_plot(self, embs, labs)
            # k-NN via streaming and centroid metrics (memory-efficient)
            metrics = self.compute_knn_streaming(embs, labs, self.hparams.knn_val_neighbors, stage)
            metrics.update(compute_centroid_metrics(embs, labs, stage))
        except Exception as e:
            log.error(f"Error during _shared_epoch_end for {stage}: {e}", exc_info=True)
            # ensure defaults on error
            metrics.setdefault(f"{stage}/knn_acc", 0.0)
            metrics.setdefault(f"{stage}/knn_balanced_acc", 0.0)
            metrics.setdefault(f"{stage}/centroid_acc", 0.0)
            metrics.setdefault(f"{stage}/centroid_balanced_acc", 0.0)
        finally:
            outputs.clear()
        return metrics

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(self.device)
        with torch.inference_mode():
            out = self(x)
        return out.cpu()