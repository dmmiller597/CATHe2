import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef

class CATHeClassifier(pl.LightningModule):
    """PyTorch Lightning module for CATH superfamily classification."""
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.5,
        learning_rate: float = 1e-5,
        use_batch_norm: bool = True,
        l1_reg: float = 1e-5,
        l2_reg: float = 1e-4
    ):
        """
        Initialize the CATH classifier.

        Args:
            embedding_dim (int): Dimension of input protein embeddings.
            hidden_sizes (List[int]): List of hidden layer sizes.
            num_classes (int): Number of CATH superfamily classes.
            dropout (float): Dropout probability.
            learning_rate (float): Learning rate for optimizer.
            use_batch_norm (bool): Whether to use batch normalization.
            l1_reg (float): L1 regularization strength.
            l2_reg (float): L2 regularization strength.
        """
        super().__init__()
        self.save_hyperparameters()

        # Build layers
        layers = []
        input_dim = embedding_dim        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity(),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, num_classes))
        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize metrics for each stage
        metrics = {
            'acc': Accuracy,
            'f1': F1Score,
            'mcc': MatthewsCorrCoef,
            'balanced_acc': lambda: Accuracy(average='macro')
        }
        for stage in ['train', 'val', 'test']:
            for name, metric_cls in metrics.items():
                setattr(self, f"{stage}_{name}", metric_cls(task="multiclass", num_classes=num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

    def _compute_regularization(self) -> torch.Tensor:
        """Compute combined L1 and L2 regularization losses."""
        l1_loss = sum(torch.sum(torch.abs(p)) for p in self.parameters() if p.requires_grad)
        l2_loss = sum(torch.sum(p ** 2) for p in self.parameters() if p.requires_grad)
        return self.hparams.l1_reg * l1_loss + self.hparams.l2_reg * l2_loss

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
        # Add L1 and L2 regularization loss
        if self.hparams.l1_reg > 0 or self.hparams.l2_reg > 0:
            loss += self._compute_regularization()

        preds = torch.argmax(logits, dim=1)
        # Update and log metrics
        metrics = {
            'loss': loss,
            'acc': self.train_acc(preds, y),
            'f1': self.train_f1(preds, y),
            'mcc': self.train_mcc(preds, y),
            'balanced_acc': self.train_balanced_acc(preds, y)
        }
        self.log_dict({f"train_{k}": v for k, v in metrics.items()},
                      prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Validation step.

        Args:
            batch (tuple): Tuple of (embeddings, labels).
            batch_idx (int): Current batch index.
            
        Returns:
            dict: Dictionary containing loss and metrics.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = {
            'loss': loss,
            'acc': self.val_acc(preds, y),
            'f1': self.val_f1(preds, y),
            'mcc': self.val_mcc(preds, y),
            'balanced_acc': self.val_balanced_acc(preds, y)
        }
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        return {'loss': loss, 'preds': preds, 'targets': y}

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Test step.

        Args:
            batch (tuple): Tuple of (embeddings, labels).
            batch_idx (int): Current batch index.
            
        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log_dict({
            "test_loss": loss,
            "test_acc": self.test_acc(preds, y),
            "test_f1": self.test_f1(preds, y)
        })
        return loss

    def configure_optimizers(self) -> dict:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            dict: Dictionary containing optimizer and lr scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "frequency": 1
            }
        } 