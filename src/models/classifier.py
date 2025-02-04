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
        
        # Initialize lists to store predictions and targets
        self.validation_step_outputs = []

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

        # Initialize metrics once and reuse them
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)
        self.mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        self.balanced_acc = Accuracy(task="multiclass", num_classes=num_classes, average='macro')

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
        
        if self.hparams.l1_reg > 0 or self.hparams.l2_reg > 0:
            loss += self._compute_regularization()
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.accuracy(preds, y), prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Validation step.

        Args:
            batch (tuple): Tuple of (embeddings, labels).
            batch_idx (int): Current batch index.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Store predictions and targets for epoch end
        self.validation_step_outputs.append({
            "preds": preds,
            "targets": y
        })
        
        # Log loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Compute metrics at epoch end"""
        # Stack all predictions and targets
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
        
        # Log metrics
        self.log_dict({
            "val_acc": self.accuracy(preds, targets),
            "val_f1": self.f1_score(preds, targets),
            "val_mcc": self.mcc(preds, targets),
            "val_balanced_acc": self.balanced_acc(preds, targets)
        }, prog_bar=True)
        
        # Clear saved predictions
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Test step.

        Args:
            batch (tuple): Tuple of (embeddings, labels).
            batch_idx (int): Current batch index.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        self.log_dict({
            "test_loss": loss,
            "test_acc": self.accuracy(preds, y),
            "test_f1": self.f1_score(preds, y),
            "test_mcc": self.mcc(preds, y),
            "test_balanced_acc": self.balanced_acc(preds, y)
        }, prog_bar=True)

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