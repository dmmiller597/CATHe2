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
import csv

from utils import get_logger
from losses import SupConLoss, SINCERELoss
from plotting import generate_tsne_plot, generate_umap_plot
from metrics import (
    compute_holdout_metrics,
    compute_centroid_metrics,
    compute_knn_metrics,
    compute_centroid_metrics_reference,
    compute_knn_metrics_reference,
)
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
        evaluated_ids_output_dir: str = "results/evaluated_ids",
        min_class_size_for_eval: int = 2,
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

        # Ensure output directory exists
        os.makedirs(self.hparams.evaluated_ids_output_dir, exist_ok=True)

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

    def validation_step(self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int) -> None:
        emb, labels, sequence_ids = batch
        with torch.inference_mode():
            proj = self(emb)
            loss = self.loss_fn(proj, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_outputs.append({
            "embeddings": proj.detach(),
            "labels": labels.detach(),
            "sequence_ids": sequence_ids
        })

    def test_step(self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int) -> None:
        emb, labels, sequence_ids = batch
        with torch.inference_mode():
            proj = self(emb)
        self._test_outputs.append({
            "embeddings": proj.cpu().detach(),
            "labels": labels.cpu().detach(),
            "sequence_ids": sequence_ids
        })

    def on_validation_epoch_end(self) -> None:
        # Compute and log validation metrics using the shared logic
        # Centroid metrics will be computed, kNN will be skipped for 'val' stage
        m = self._shared_epoch_end(self._val_outputs, 'val')
        if m:
            # filter out NaN metrics
            loggable = {k: v for k, v in m.items() if not (isinstance(v, float) and np.isnan(v))}
            self.log_dict(loggable, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # Outputs are cleared within _shared_epoch_end

    def on_test_epoch_end(self) -> None:
        # Compute and log test metrics using the shared logic
        # Both Centroid and kNN metrics will be computed for 'test' stage
        m = self._shared_epoch_end(self._test_outputs, 'test')
        if m:
            # filter out NaN metrics
            loggable = {k: v for k, v in m.items() if not (isinstance(v, float) and np.isnan(v))}
            self.log_dict(loggable, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # Outputs are cleared within _shared_epoch_end

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

    def _shared_epoch_end(self, outputs: List[Dict[str, Any]], stage: str) -> Dict[str, float]:
        """
        Common logic for epoch ends. Gathers data, computes metrics, and saves
        evaluated sequence IDs for the test stage.
        """
        metrics: Dict[str, float] = {}
        # Indices of samples evaluated by centroid metric (should be same as kNN)
        evaluated_indices: Optional[Tensor] = None

        if not outputs:
            log.warning(f"No outputs provided for stage={stage} metrics computation.")
            # Return default zero metrics
            for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                metrics[f"{stage}/centroid_{name}"] = 0.0
            if stage == 'test': # Only add default KNN for test stage if outputs are empty
                for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                    metrics[f"{stage}/knn_1_{name}"] = 0.0 # k=1
            outputs.clear() # Still clear outputs even if empty
            return metrics

        try:
            # Gather embeddings, labels, and sequence IDs
            embs_cpu = torch.cat([o['embeddings'].cpu().detach() for o in outputs])
            labs_cpu = torch.cat([o['labels'].cpu().detach() for o in outputs])
            # Sequence IDs are lists of strings, flatten them
            ids_list: List[str] = [seq_id for o in outputs for seq_id in o['sequence_ids']]

            # --- Visualization (only during validation, every 5 epochs if enabled) ---
            if stage == 'val' and self.hparams.enable_visualization and self.current_epoch % 5 == 0:
                log.info(f"Generating '{self.hparams.visualization_method}' plot for validation epoch {self.current_epoch}...")
                if self.hparams.visualization_method == "tsne":
                    generate_tsne_plot(self, embs_cpu, labs_cpu)
                elif self.hparams.visualization_method == "umap":
                    generate_umap_plot(self, embs_cpu, labs_cpu)
                else:
                    log.warning(f"Unsupported visualization method '{self.hparams.visualization_method}' specified, skipping plot generation.")
            # --- End Visualization ---

            # 1. Compute Centroid metrics (receives indices)
            centroid_start_time = time.time()
            centroid_metrics, evaluated_indices = compute_centroid_metrics(
                embs_cpu, labs_cpu, stage, min_class_size=self.hparams.min_class_size_for_eval
            )
            metrics.update(centroid_metrics)
            centroid_elapsed = time.time() - centroid_start_time
            log.info(f"Centroid metrics computed in {centroid_elapsed:.2f} seconds for {stage}.")

            # 2. Compute k-NN metrics (Only for test stage, ignore returned indices)
            if stage == 'test':
                knn_start_time = time.time()
                knn_metrics, _ = compute_knn_metrics(
                    embs_cpu, labs_cpu,
                    k=1, stage=stage, knn_batch_size=self.hparams.knn_batch_size,
                    min_class_size=self.hparams.min_class_size_for_eval
                )
                metrics.update(knn_metrics)
                knn_elapsed = time.time() - knn_start_time
                log.info(f"k-NN (k=1) metrics computed in {knn_elapsed:.2f} seconds for {stage}.")

            # --- ADDED: Save evaluated IDs for the test stage using indices ---
            if stage == 'test' and evaluated_indices is not None and len(evaluated_indices) > 0:
                try:
                    # Get label decoder from datamodule
                    label_decoder = self.trainer.datamodule.get_label_decoder()

                    # Select the IDs and labels corresponding to the evaluated indices
                    eval_ids = [ids_list[i] for i in evaluated_indices.tolist()]
                    eval_true_labels = labs_cpu[evaluated_indices]
                    eval_sfs = [label_decoder.get(lbl.item(), "N/A") for lbl in eval_true_labels]

                    # Define output filename
                    output_filename = os.path.join(
                        self.hparams.evaluated_ids_output_dir,
                        f"test_evaluated_ids_epoch_{self.current_epoch}.csv"
                    )
                    # Write to CSV
                    with open(output_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['sequence_id', 'true_superfamily']) # Header
                        writer.writerows(zip(eval_ids, eval_sfs))
                    log.info(f"Saved {len(eval_ids)} evaluated sequence IDs for test stage to {output_filename}")

                except AttributeError:
                     log.error("Could not get label decoder, possibly datamodule not available. Skipping saving IDs.")
                except Exception as e_save:
                     log.error(f"Failed to save evaluated sequence IDs for test stage: {e_save}", exc_info=True)
            elif stage == 'test':
                log.warning("No evaluated indices available for test stage, skipping ID saving.")
            # --- END ADDED ---

        except Exception as e:
            log.error(f"Error during _shared_epoch_end for {stage}: {e}", exc_info=True)
            # Ensure defaults on error
            for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                metrics.setdefault(f"{stage}/centroid_{name}", 0.0)
            if stage == 'test': # Only set default KNN for test stage on error
                for name in ("acc", "balanced_acc", "precision", "recall", "f1_macro"):
                    metrics.setdefault(f"{stage}/knn_1_{name}", 0.0)
        finally:
            outputs.clear() # Clear outputs after processing

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
        # This is no longer strictly necessary if we only compute self-classification metrics,
        # but keep it in case reference metrics are needed later or by other parts.
        # Can be commented out if reference set is truly unused.
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