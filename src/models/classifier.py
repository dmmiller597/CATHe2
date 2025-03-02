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
        
        # Loss function (kept on GPU for efficiency)
        self.criterion = nn.CrossEntropyLoss()
        
        # Simple training metrics can stay on GPU as they're lightweight
        # and used frequently during training
        self.train_loss = MeanMetric()
        
        # Complex metrics with high memory requirements moved to CPU
        # These are primarily used for evaluation, not training decisions
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes).to('cpu')
        
        # Create metrics collection and move to CPU
        self.val_metrics = MetricCollection({
            'acc': Accuracy(task="multiclass", num_classes=num_classes),
            'balanced_acc': Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        }, prefix='val/').to('cpu')
        
        self.test_metrics = self.val_metrics.clone(prefix='test/').to('cpu')
        self.val_acc_best = MaxMetric().to('cpu')
    
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
        
        # Update loss metric on GPU (lightweight operation)
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Move predictions and targets to CPU for complex metrics calculation
        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()
        self.train_acc.update(preds_cpu, targets_cpu)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        loss, preds, targets = self.model_step(batch)
        
        # Log loss directly from GPU
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Move predictions and targets to CPU for complex metrics calculation
        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()
        self.val_metrics.update(preds_cpu, targets_cpu)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, preds, targets = self.model_step(batch)
        
        # Log loss directly from GPU
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        
        # Move predictions and targets to CPU for complex metrics calculation
        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()
        self.test_metrics.update(preds_cpu, targets_cpu)

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
        self.train_loss.reset()
        self.train_acc.reset()

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
        