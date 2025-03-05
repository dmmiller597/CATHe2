import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict, Any, Tuple
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, MetricCollection, MeanMetric, MaxMetric
import torch.nn.functional as F

class CATHeClassifier(pl.LightningModule):
    """PyTorch Lightning module for CATH superfamily classification."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler: Dict[str, Any]
    ):
        """
        Initialize the CATH classifier.

        Args:
            embedding_dim: Dimension of input protein embeddings
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of CATH superfamily classes
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization) for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        # Build layers
        layers = []
        in_features = embedding_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size, bias=True),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)
        self._init_weights()

        # Define metrics for testing only
        self.test_metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "balanced_acc": Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
        })

        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def _init_weights(self) -> None:
        """Initialize network weights using Kaiming initialization with GELU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Training step - compute loss."""
        loss, preds, targets = self.model_step(batch)
        
        # Update loss
        self.train_loss(loss)
        
        # Log loss
        self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=False)
        
        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step - compute loss."""
        loss, preds, targets = self.model_step(batch)
        
        # Update loss
        self.val_loss(loss)
        
        # Log loss
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step - compute metrics."""
        loss, preds, targets = self.model_step(batch)
        self.test_metrics(preds, targets)

    def on_train_epoch_end(self) -> None:
        """Handle training epoch end - log loss and reset."""
        self.log('train/loss_epoch', self.train_loss.compute(), prog_bar=True)
        self.train_loss.reset()

    def on_validation_epoch_end(self) -> None:
        """Handle validation epoch end - log loss and reset."""
        self.log('val/loss', self.val_loss.compute(), prog_bar=True)
        self.val_loss.reset()

    def on_test_epoch_end(self) -> None:
        """Handle test epoch end - log metrics and reset."""
        test_metrics = {f"test/{k}": v for k, v in self.test_metrics.compute().items()}
        self.log_dict(test_metrics, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.lr_scheduler["mode"],
            factor=self.hparams.lr_scheduler["factor"],
            patience=self.hparams.lr_scheduler["patience"],
            min_lr=self.hparams.lr_scheduler["min_lr"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.lr_scheduler["monitor"],
                "interval": "epoch"
            }
        }