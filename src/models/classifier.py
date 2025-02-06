import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, FocalLoss

class CATHeClassifier(pl.LightningModule):
    """PyTorch Lightning module for CATH superfamily classification."""
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
    ):
        """
        Initialize the CATH classifier.

        Args:
            embedding_dim: Dimension of input protein embeddings
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of CATH superfamily classes
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler_factor: Factor by which to reduce LR on plateau
            scheduler_patience: Number of epochs to wait before reducing LR
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Build MLP layers
        layers = []
        in_features = embedding_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)
        
        # Standard metrics setup
        self.criterion = FocalLoss(
            task="multiclass",
            num_classes=num_classes,
            gamma=2.0
        )
        self._setup_metrics(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (tuple): Tuple of (embeddings, labels).
            batch_idx (int): Current batch index.
            
        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.accuracy(logits.argmax(1), y), 
                prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)
        
        # Update metrics
        self.accuracy.update(preds, y)
        self.f1_score.update(preds, y)
        self.mcc.update(preds, y)
        self.balanced_acc.update(preds, y)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        metrics = {
            "val_acc": self.accuracy.compute().float(),
            "val_f1": self.f1_score.compute().float(),
            "val_mcc": self.mcc.compute().float(),
            "val_balanced_acc": self.balanced_acc.compute().float()
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        
        # Reset metrics
        for metric in [self.accuracy, self.f1_score, self.mcc, self.balanced_acc]:
            metric.reset()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        preds = logits.argmax(1)
        
        # Update metrics
        self.accuracy.update(preds, y)
        self.f1_score.update(preds, y)
        self.mcc.update(preds, y)
        self.balanced_acc.update(preds, y)

    def on_test_epoch_end(self) -> None:
        metrics = {
            "test_acc": self.accuracy.compute().float(),
            "test_f1": self.f1_score.compute().float(),
            "test_mcc": self.mcc.compute().float(),
            "test_balanced_acc": self.balanced_acc.compute().float()
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        
        # Reset metrics
        for metric in [self.accuracy, self.f1_score, self.mcc, self.balanced_acc]:
            metric.reset()

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_balanced_acc",
                "frequency": 1
            }
        }

    def _setup_metrics(self, num_classes: int):
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)
        self.mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        self.balanced_acc = Accuracy(task="multiclass", num_classes=num_classes, average='macro') 