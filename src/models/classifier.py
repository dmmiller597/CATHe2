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
        dropout: float = 0.5,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        lr_scheduler: Dict[str, Any] = None
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

        # Initialize metrics using MetricCollection consistently
        # Move metrics to CPU and compute only at epoch end for better performance
        metrics = {
            'acc': Accuracy(task="multiclass", num_classes=num_classes),
            'balanced_acc': Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
            'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro')
        }
        
        # Use consistent MetricCollection pattern for all stages
        self.train_metrics = MetricCollection(metrics, prefix='train/')
        self.val_metrics = MetricCollection(metrics, prefix='val/')
        self.test_metrics = MetricCollection(metrics, prefix='test/')
        
        # Move metrics to CPU to save GPU memory
        self.train_metrics.to('cpu')
        self.val_metrics.to('cpu')
        self.test_metrics.to('cpu')
        
        # Track best performance
        self.val_acc_best = MaxMetric()
        
        # Loss tracking for efficiency
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
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
        
        # Forward pass
        logits = self(x)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Get predictions (argmax) - keep on same device for metrics
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - compute loss and update metrics."""
        loss, preds, targets = self.model_step(batch)
        
        # Track loss with MeanMetric instead of logging every step
        self.train_loss(loss)
        
        # Only log loss on step for progress bar, not for history
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        
        # Detach and move to CPU before updating metrics to save GPU memory
        # Only update metrics every 50 batches to improve speed
        if batch_idx % 50 == 0:
            self.train_metrics(preds.detach().cpu(), targets.detach().cpu())
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step - compute loss and update metrics."""
        loss, preds, targets = self.model_step(batch)
        
        # Track loss with MeanMetric
        self.val_loss(loss)
        
        # Detach and move to CPU before updating metrics
        self.val_metrics(preds.detach().cpu(), targets.detach().cpu())

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step - compute loss and update metrics."""
        loss, preds, targets = self.model_step(batch)
        
        # Track loss with MeanMetric
        self.test_loss(loss)
        
        # Detach and move to CPU before updating metrics
        self.test_metrics(preds.detach().cpu(), targets.detach().cpu())

    def on_train_epoch_end(self) -> None:
        """Handle training epoch end - log metrics and reset."""
        # Log the mean loss for the epoch
        self.log('train/loss_epoch', self.train_loss.compute(), prog_bar=True)
        self.train_loss.reset()
        
        # Compute and log metrics
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Handle validation epoch end - log metrics and reset."""
        # Log the mean loss for the epoch
        self.log('val/loss', self.val_loss.compute(), prog_bar=True)
        self.val_loss.reset()
        
        # Compute and log metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        
        # Update and log best accuracy
        self.val_acc_best.update(metrics['val/balanced_acc'])
        self.log("val/balanced_acc_best", self.val_acc_best.compute(), prog_bar=True)
        
        # Reset metrics
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Handle test epoch end - log metrics and reset."""
        # Log the mean loss for the epoch
        self.log('test/loss', self.test_loss.compute())
        self.test_loss.reset()
        
        # Compute and log metrics
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
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