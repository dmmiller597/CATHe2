import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict, Any, Tuple
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, MetricCollection, Metric
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
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
        self._init_weights()  # Initialize weights
        
        # Initialize FocalLoss with label smoothing
        self.criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        
        # Training metric
        self.train_acc = Accuracy(task="multiclass", 
                                num_classes=num_classes,
                                average='micro')
        
        # Evaluation metrics
        eval_metrics = {
            'acc': Accuracy(task="multiclass", num_classes=num_classes, average='micro'),
            'balanced_acc': Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
            'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
            'mcc': MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        }
        
        # Create separate collections for validation and test
        self.val_metrics = MetricCollection(eval_metrics).clone(prefix='val_')
        self.test_metrics = MetricCollection(eval_metrics).clone(prefix='test_')
    
    def _init_weights(self) -> None:
        """Initialize network weights using Kaiming initialization."""
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
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute training step and return loss."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        
        # Compute and log metrics
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Execute validation step and log metrics."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        
        # Update all metrics at once
        metric_dict = self.val_metrics(preds, y)
        
        # Log metrics individually
        self.log("val_loss", loss, on_epoch=True)
        for name, value in metric_dict.items():
            self.log(name, value, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Execute test step and log metrics."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        
        # Update all metrics at once
        metric_dict = self.test_metrics(preds, y)
        
        # Log metrics individually
        self.log("test_loss", loss, on_epoch=True)
        for name, value in metric_dict.items():
            self.log(name, value, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        """Reset training metrics."""
        self.train_acc.reset()

    def on_validation_epoch_end(self) -> None:
        """Compute final validation metrics and reset."""
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Compute final test metrics and reset."""
        metrics = self.test_metrics.compute()
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
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
        