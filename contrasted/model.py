import os
from typing import Any, Dict, List, Optional, Tuple

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR

from utils import get_logger
from losses import SupConLoss, SINCERELoss
from plotting import generate_tsne_plot, generate_umap_plot
from metrics import compute_centroid_metrics, compute_knn_metrics, compute_centroid_metrics_reference, compute_knn_metrics_reference
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
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        current = h
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)



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
        enable_visualization: bool = True,
        tsne_viz_dir: str = "results/tsne_plots",
        umap_viz_dir: str = "results/umap_plots",
        temperature: float = 0.07,
        seed: int = 42,
        knn_batch_size: int = 1024,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Validations
        if not (0 < self.hparams.warmup_start_factor <= 1.0):
            raise ValueError("warmup_start_factor must be > 0 and <= 1.0")
        if self.hparams.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if self.hparams.enable_visualization and self.hparams.visualization_method not in ("umap", "tsne"):
            raise ValueError("visualization_method must be 'umap' or 'tsne'.")

        # Model components
        self.projection = build_projection_network(
            input_dim=self.hparams.input_embedding_dim,
            hidden_dims=self.hparams.projection_hidden_dims,
            output_dim=self.hparams.output_embedding_dim,
            dropout=self.hparams.dropout,
            use_layer_norm=self.hparams.use_layer_norm,
        )

        # Supervised contrastive loss (drop‐in replacement)
        self.loss_fn = SupConLoss(temperature=self.hparams.temperature)

        # Buffers for metrics
        self._val_outputs: List[Dict[str, Tensor]] = []
        self._test_outputs: List[Dict[str, Tensor]] = []
        # Buffers for reference (training) set:
        self._ref_embs: Optional[Tensor] = None
        self._ref_labels: Optional[Tensor] = None
        self._ref_built: bool = False

        # Ensure viz dirs exist if enabled
        if self.hparams.enable_visualization:
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
            self.log_dict(loggable, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        m = self._shared_epoch_end(self._test_outputs, 'test')
        if m:
            # filter out NaN metrics
            loggable = {k: v for k, v in m.items() if not (isinstance(v, float) and np.isnan(v))}
            self.log_dict(loggable, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

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
            # Add defaults for kNN metrics too
            for k in [1]:
                for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                    metrics[f"{stage}/knn_{k}_{name}"] = 0.0
            outputs.clear()
            return metrics
        try:
            embs_cpu = torch.cat([o['embeddings'].cpu() for o in outputs])
            labs_cpu = torch.cat([o['labels'].cpu()    for o in outputs])

            if stage == 'test':
                # centroid & kNN against TRAINING‐set cache
                metrics.update(
                    compute_centroid_metrics_reference(
                        embs_cpu, labs_cpu, self._ref_embs, self._ref_labels, stage
                    )
                )
                metrics.update(
                    compute_knn_metrics_reference(
                        embs_cpu, labs_cpu,
                        self._ref_embs, self._ref_labels,
                        1, stage, self.hparams.knn_batch_size
                    )
                )
                metrics.update(
                    compute_knn_metrics_reference(
                        embs_cpu, labs_cpu,
                        self._ref_embs, self._ref_labels,
                        3, stage, self.hparams.knn_batch_size
                    )
                )
            else:
                # original self‐classification (val or sanity)
                metrics.update(compute_centroid_metrics(embs_cpu, labs_cpu, stage))
                if (
                    stage == 'test'
                    or (stage == 'val' and False)  # keep your existing val‐knn logic
                ):
                    knn_batch_size = self.hparams.knn_batch_size
                    log.info(f"Computing KNN metrics for {stage} at epoch {getattr(self, 'current_epoch', 'N/A')}")
                    knn_start_time = time.time()
                    metrics.update(compute_knn_metrics(embs_cpu, labs_cpu, 1, stage, knn_batch_size))
                    metrics.update(compute_knn_metrics(embs_cpu, labs_cpu, 3, stage, knn_batch_size))
                    knn_elapsed = time.time() - knn_start_time
                    log.info(f"KNN metrics computed in {knn_elapsed:.2f} seconds for {stage}.")

            # optional visualization
            if (
                stage == 'val'
                and hasattr(self, 'trainer')
                and not self.trainer.sanity_checking
                and self.current_epoch > 0
                and self.current_epoch % 10 == 0
                and self.hparams.enable_visualization
            ):

                log.info(f"Visualizing embeddings (method={self.hparams.visualization_method}, epoch={self.current_epoch}, n={embs_cpu.size(0)})")
                start_time = time.time()
                if self.hparams.visualization_method == 'umap':
                    generate_umap_plot(self, embs_cpu, labs_cpu)
                else:
                    generate_tsne_plot(self, embs_cpu, labs_cpu)
                elapsed = time.time() - start_time
                log.info(f"Visualization completed in {elapsed:.2f} seconds.")
            
        except Exception as e:
            log.error(f"Error during _shared_epoch_end for {stage}: {e}", exc_info=True)
            # ensure defaults on error
            metrics.setdefault(f"{stage}/centroid_acc", 0.0)
            metrics.setdefault(f"{stage}/centroid_balanced_acc", 0.0)
            metrics.setdefault(f"{stage}/centroid_precision", 0.0)
            metrics.setdefault(f"{stage}/centroid_recall", 0.0)
            metrics.setdefault(f"{stage}/centroid_f1_macro", 0.0)
            # Add defaults for kNN metrics too
            for k in [1]:
                 for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                    metrics.setdefault(f"{stage}/knn_{k}_{name}", 0.0)
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

    def on_test_epoch_start(self) -> None:
        """Build the training‐set embedding/label cache once before any test metrics."""
        if not self._ref_built:
            self._build_reference_set()
            self._ref_built = True

    def _build_reference_set(self) -> None:
        """Run forward over the train loader to collect all normalized embeddings + labels."""
        log.info("Building reference embeddings from train set")
        train_loader = self.trainer.datamodule.train_dataloader()
        all_embs, all_labels = [], []
        self.eval()
        for emb, labels in train_loader:
            # move embeddings to same device as the model (GPU/CPU)
            emb = emb.to(self.device)
            with torch.inference_mode():
                proj = self(emb)
            all_embs.append(proj.cpu().detach())
            all_labels.append(labels.cpu().detach())
        self.train()
        self._ref_embs = torch.cat(all_embs, dim=0)
        self._ref_labels = torch.cat(all_labels, dim=0)