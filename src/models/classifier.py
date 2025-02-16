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
            weight_decay: Weight decay (L2 regularization) for optimizer
            scheduler_factor: Factor by which to reduce LR on plateau
            scheduler_patience: Number of epochs to wait before reducing LR
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
        
        # Initialize metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_metrics = MetricCollection({
            'acc': Accuracy(task="multiclass", num_classes=num_classes),
            'balanced_acc': Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
            'f1': F1Score(task="multiclass", num_classes=num_classes),
            'mcc': MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        }, prefix='val/')
        self.test_metrics = self.val_metrics.clone(prefix='test/')
        self.val_acc_best = MaxMetric()
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
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, targets = self.model_step(batch)
        
        # Log only loss and accuracy during training
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/acc', self.train_acc(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        loss, preds, targets = self.model_step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(preds, targets)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, preds, targets = self.model_step(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.test_metrics.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        """Handle validation epoch end."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.val_acc_best.update(metrics['val/balanced_acc'])
        self.log("val/balanced_acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Handle test epoch end."""
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        """Reset training metrics at epoch end."""
        self.train_acc.reset()

    def configure_optimizers(self):
        """Configure optimizer only - scheduler handled by callbacks."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        