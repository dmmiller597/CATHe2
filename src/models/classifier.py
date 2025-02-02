import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Optional, Dict
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class CATHeClassifier(pl.LightningModule):
    """PyTorch Lightning module for CATH superfamily classification."""
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        use_batch_norm: bool = True
    ):
        """Initialize the CATH classifier.
        
        Args:
            embedding_dim: Dimension of input protein embeddings
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of CATH superfamily classes
            dropout: Dropout probability (default: 0.2)
            learning_rate: Learning rate for optimization (default: 0.001)
            use_batch_norm: Whether to use batch normalization (default: True)
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
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_size
            
        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)
        
        # Confusion matrix for validation
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store predictions for epoch end analysis
        self.validation_step_outputs = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Tuple of (embeddings, labels)
            batch_idx: Index of current batch
            
        Returns:
            Loss tensor
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate and log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        f1 = self.train_f1(preds, y)
        
        # Log metrics
        self.log_dict({
            "train_loss": loss,
            "train_acc": acc,
            "train_f1": f1
        }, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (embeddings, labels)
            batch_idx: Index of current batch
            
        Returns:
            Loss tensor
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        f1 = self.val_f1(preds, y)
        
        # Update confusion matrix
        self.val_confmat.update(preds, y)
        
        # Log metrics
        self.log_dict({
            "val_loss": loss,
            "val_acc": acc,
            "val_f1": f1
        }, prog_bar=True)
        
        # Store for epoch end analysis
        self.validation_step_outputs.append({
            'val_loss': loss,
            'preds': preds,
            'targets': y
        })
        
        return loss

    def on_validation_epoch_end(self):
        """Compute and log epoch-level validation metrics."""
        # Compute confusion matrix
        conf_mat = self.val_confmat.compute()
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_mat.cpu().numpy(), annot=True, fmt='d', cmap='Blues')
        plt.title('Validation Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Log figure to tensorboard
        if self.logger:
            self.logger.experiment.add_figure(
                'confusion_matrix',
                plt.gcf(),
                global_step=self.current_epoch
            )
        
        plt.close()
        
        # Reset validation step outputs
        self.validation_step_outputs.clear()
        
        # Reset confusion matrix
        self.val_confmat.reset()

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step.
        
        Args:
            batch: Tuple of (embeddings, labels)
            batch_idx: Index of current batch
            
        Returns:
            Loss tensor
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate and log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        f1 = self.test_f1(preds, y)
        
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
            "test_f1": f1
        })
        
        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Dict containing optimizer and lr scheduler configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        } 