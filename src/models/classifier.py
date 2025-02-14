import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict, Any
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, MetricCollection, Metric
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        """
        Initialize Focal Loss.
        
        Args:
            gamma: Focus parameter that modulates loss for hard examples (default: 2.0)
            label_smoothing: Label smoothing factor (default: 0.1)
        """
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert targets to one-hot with smoothing
        num_classes = inputs.shape[1]
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1
            )
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
        
        # Compute focal loss with label smoothing
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(targets_smooth * log_probs).sum(dim=1)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

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
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
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
            scheduler_factor: Factor by which to reduce LR on plateau
            scheduler_patience: Number of epochs to wait before reducing LR
            focal_gamma: Focus parameter for focal loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Build layers
        layers = []
        in_features = embedding_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(
                    in_features, 
                    hidden_size,
                    bias=True
                ),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size
            
        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)
        
        # Initialize FocalLoss with label smoothing
        self.criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        
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

    def _setup_metrics(self, num_classes: int) -> None:
        """Initialize metrics for training, validation and testing."""
        metrics = MetricCollection({
            'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
            'f1_score': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
            'mcc': MatthewsCorrCoef(task="multiclass", num_classes=num_classes),
            'balanced_acc': Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        })
        
        # Create separate metric instances for each stage
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def _log_metrics(self, metrics: MetricCollection, loss: torch.Tensor, stage: str) -> None:
        """Centralized logging function for metrics and loss."""
        on_step = stage == 'train'  # Only log steps during training
        sync_dist = stage != 'train'  # Sync metrics across GPUs for val/test
        
        # Log loss
        self.log(
            f"{stage}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist
        )
        
        # Log metrics
        self.log_dict(
            metrics,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist
        )

    def _step(self, batch: tuple, stage: str) -> torch.Tensor:
        """Unified step function for training, validation and testing."""
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        
        # Select appropriate metrics
        metrics = getattr(self, f"{stage}_metrics")
        metrics.update(preds, y)
        
        # Log metrics
        self._log_metrics(metrics, loss, stage)
        
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "test")

    def _epoch_end(self, stage: str) -> None:
        """Unified epoch end function for all stages."""
        metrics = getattr(self, f"{stage}_metrics")
        self.log_dict(
            metrics.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=stage != 'train'
        )
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._epoch_end('train')

    def on_validation_epoch_end(self) -> None:
        self._epoch_end('val')

    def on_test_epoch_end(self) -> None:
        self._epoch_end('test')

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
            patience=self.hparams.scheduler_patience,
            min_lr=1e-8
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_balanced_acc",
                "frequency": 1
            }
        } 