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
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        visualization_method: str = "tsne",
        tsne_viz_dir: str = "results/tsne_plots",
        umap_viz_dir: str = "results/umap_plots",
        temperature: float = 0.07,
        seed: int = 42,
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
            metrics[f"{stage}/centroid_acc"] = 0.0
            metrics[f"{stage}/centroid_balanced_acc"] = 0.0
            metrics[f"{stage}/centroid_precision"] = 0.0
            metrics[f"{stage}/centroid_recall"] = 0.0
            metrics[f"{stage}/centroid_f1_macro"] = 0.0
            outputs.clear()
            return metrics
        try:
            # Concatenate and move data to CPU before centroid computation to avoid GPU OOM
            embs_cpu = torch.cat([o['embeddings'].detach().cpu() for o in outputs])
            labs_cpu = torch.cat([o['labels'].detach().cpu() for o in outputs])
            # optional visualization
            if (
                stage == 'val'
                and hasattr(self, 'trainer')
                and not self.trainer.sanity_checking
                and self.current_epoch > 0
                and self.current_epoch % 10 == 0
            ):
                log.info(f"Visualizing embeddings (method={self.hparams.visualization_method}, epoch={self.current_epoch}, n={embs_cpu.size(0)})")
                if self.hparams.visualization_method == 'umap':
                    generate_umap_plot(self, embs_cpu, labs_cpu)
                else:
                    generate_tsne_plot(self, embs_cpu, labs_cpu)
            # troid metrics (memory-efficient)
            metrics.update(compute_centroid_metrics(embs_cpu, labs_cpu, stage))
        except Exception as e:
            log.error(f"Error during _shared_epoch_end for {stage}: {e}", exc_info=True)
            # ensure defaults on error
            metrics.setdefault(f"{stage}/centroid_acc", 0.0)
            metrics.setdefault(f"{stage}/centroid_balanced_acc", 0.0)
            metrics.setdefault(f"{stage}/centroid_precision", 0.0)
            metrics.setdefault(f"{stage}/centroid_recall", 0.0)
            metrics.setdefault(f"{stage}/centroid_f1_macro", 0.0)
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