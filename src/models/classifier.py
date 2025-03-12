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
            lr_scheduler: Learning rate scheduler configuration
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

        # Define metrics - for both training and evaluation
        metrics = {
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "balanced_acc": Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
        }
        
        # Create separate metric collections for different stages
        self.train_metrics = MetricCollection(metrics, prefix='train_')
        self.val_metrics = MetricCollection(metrics, prefix='val_')
        self.test_metrics = MetricCollection(metrics, prefix='test_')
        
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
        return logits, loss, preds

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits, loss, preds = self.model_step(batch)
        
        # Update metrics
        self.train_loss(loss)
        self.train_metrics.update(preds, y)
        
        # Log loss per step (helpful for debugging)
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits, loss, preds = self.model_step(batch)
        
        # Update metrics
        self.val_loss(loss)
        self.val_metrics.update(preds, y)
        
        # Log loss per step
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits, loss, preds = self.model_step(batch)
        
        # Update metrics
        self.test_metrics.update(preds, y)

    def on_train_epoch_end(self) -> None:
        # Compute epoch metrics
        train_loss = self.train_loss.compute()
        metrics = self.train_metrics.compute()
        
        # Log all metrics
        self.log("train/loss_epoch", train_loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"train/{name}_epoch", value, prog_bar=True)
        
        # Reset metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        # Compute epoch metrics
        val_loss = self.val_loss.compute()
        metrics = self.val_metrics.compute()
        
        # Log all metrics
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=True, sync_dist=True)
        
        # Reset metrics
        self.val_loss.reset()
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        # Log all test metrics
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value, sync_dist=True)
        
        # Reset metrics
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer with warmup and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create warmup + plateau scheduler using LambdaLR + ReduceLROnPlateau
        warmup_epochs = 5  # Number of epochs for warmup phase
        
        # Step 1: Define lambda function for warmup
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup from 10% to 100% of base lr
                return 0.1 + 0.9 * (epoch / warmup_epochs)
            return 1.0  # After warmup, return full lr multiplier (1.0)
        
        # Step 2: Create the warmup scheduler that affects base lr
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda
        )
        
        # Step 3: Create ReduceLROnPlateau for after warmup
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.lr_scheduler["mode"],
            factor=self.hparams.lr_scheduler["factor"],
            patience=self.hparams.lr_scheduler["patience"],
            min_lr=self.hparams.lr_scheduler["min_lr"]
        )
        
        # Return a sequential scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # First apply warmup on epoch level
                "scheduler": warmup_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "warmup"
            },
            # Then PyTorch Lightning will automatically switch to ReduceLROnPlateau
            # after warmup_epochs are completed
            "reduce_on_plateau": {
                "scheduler": plateau_scheduler,
                "monitor": self.hparams.lr_scheduler["monitor"],
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": "plateau"
            }
        }