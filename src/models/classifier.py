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

        # Memory-efficient metrics setup
        # For training, track basic accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # For validation, use sync_on_compute=True to handle device issues automatically
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, sync_on_compute=True)
        self.val_balanced_acc = Accuracy(task="multiclass", num_classes=num_classes, average='macro', sync_on_compute=True)
        
        # Track best performance
        self.val_balanced_acc_best = MaxMetric()
        
        # Loss tracking for efficiency
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        # For test, configure metrics with sync_on_compute
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, sync_on_compute=True)
        self.test_balanced_acc = Accuracy(task="multiclass", num_classes=num_classes, average='macro', sync_on_compute=True)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro', sync_on_compute=True)
        self.test_mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes, sync_on_compute=True)
        
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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Training step - compute loss and update metrics."""
        loss, preds, targets = self.model_step(batch)
        
        # Always accumulate loss
        self.train_loss(loss.detach())
        
        # Log step-level loss
        self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=False)
        
        # Update accuracy
        self.train_acc(preds, targets)
        
        # Return loss for backward pass
        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step - compute loss and update metrics one by one to save memory."""
        loss, preds, targets = self.model_step(batch)
        
        # Track loss with MeanMetric
        self.val_loss(loss.detach())
        
        # Update metrics directly
        self.val_acc(preds, targets)
        self.val_balanced_acc(preds, targets)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step - compute metrics individually to avoid memory issues."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics directly
        self.test_acc(preds, targets)
        self.test_balanced_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_mcc(preds, targets)

    def on_train_epoch_end(self) -> None:
        """Handle training epoch end - log metrics and reset."""
        # Log the mean loss for the epoch
        self.log('train/loss_epoch', self.train_loss.compute(), prog_bar=True)
        self.train_loss.reset()
        
        # Compute and log only basic accuracy
        train_acc = self.train_acc.compute()
        self.log('train/acc', train_acc, prog_bar=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self) -> None:
        """Handle validation epoch end - log metrics and reset."""
        # Log the mean loss for the epoch
        self.log('val/loss', self.val_loss.compute(), prog_bar=True)
        self.val_loss.reset()
        
        # Compute and log metrics individually
        val_acc = self.val_acc.compute()
        val_balanced_acc = self.val_balanced_acc.compute()
        
        self.log('val/acc', val_acc, prog_bar=True)
        self.log('val/balanced_acc', val_balanced_acc, prog_bar=True)
        
        # Update and log best accuracy using balanced_acc
        self.val_balanced_acc_best.update(val_balanced_acc)
        self.log("val/balanced_acc_best", self.val_balanced_acc_best.compute(), prog_bar=True)
        
        # Reset metrics
        self.val_acc.reset()
        self.val_balanced_acc.reset()

    def on_test_epoch_end(self) -> None:
        """Handle test epoch end - log metrics and reset."""
        # Compute and log metrics individually
        test_acc = self.test_acc.compute()
        test_balanced_acc = self.test_balanced_acc.compute()
        test_f1 = self.test_f1.compute()
        test_mcc = self.test_mcc.compute()
        
        self.log('test/acc', test_acc)
        self.log('test/balanced_acc', test_balanced_acc)
        self.log('test/f1', test_f1)
        self.log('test/mcc', test_mcc)
        
        # Reset metrics
        self.test_acc.reset()
        self.test_balanced_acc.reset()
        self.test_f1.reset()
        self.test_mcc.reset()

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