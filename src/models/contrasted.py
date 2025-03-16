import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict, Any, Tuple, Optional
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining."""
    
    def __init__(self, margin: float = 1.0):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Calculate triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class as anchor)
            negative: Negative embeddings (different class from anchor)
            
        Returns:
            Triplet loss value
        """
        # Calculate distances
        pos_dist = torch.sum(torch.square(anchor - positive), dim=1)
        neg_dist = torch.sum(torch.square(anchor - negative), dim=1)
        
        # Basic triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class ContrastiveCATHeModel(pl.LightningModule):
    """PyTorch Lightning module for contrastive learning of CATH superfamilies."""
    
    def __init__(
        self,
        embedding_dim: int,
        projection_dims: List[int],
        output_dim: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-5,
        margin: float = 1.0,
        lr_scheduler: Dict[str, Any] = None,
        n_neighbors: int = 5
    ):
        """Initialize the contrastive learning model.
        
        Args:
            embedding_dim: Dimension of input protein embeddings
            projection_dims: List of hidden layer sizes for projection network
            output_dim: Dimension of the output embedding space
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization) for optimizer
            margin: Margin for triplet loss
            lr_scheduler: Learning rate scheduler configuration
            n_neighbors: Number of neighbors to consider for kNN classification
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Build projection network
        layers = []
        in_features = embedding_dim
        for hidden_size in projection_dims:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size
            
        # Final projection layer with L2 normalization
        layers.extend([
            nn.Linear(in_features, output_dim),
        ])
        
        self.projection = nn.Sequential(*layers)
        self._init_weights()
        
        # Loss function
        self.triplet_loss = TripletLoss(margin=margin)
        
        # For computing metrics
        self.train_loss = torch.tensor(0.0)
        self.val_loss = torch.tensor(0.0)
        
        # Store embeddings and labels for evaluation
        self.val_embeddings = []
        self.val_labels = []
        self.test_embeddings = []
        self.test_labels = []
        
        # Set up kNN classifier
        self.n_neighbors = n_neighbors
        self.knn = None

    def _init_weights(self) -> None:
        """Initialize network weights."""
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
        """Project input embeddings to the new embedding space.
        
        Args:
            x: Input protein embeddings
            
        Returns:
            Projected embeddings
        """
        embeddings = self.projection(x)
        # L2 normalize the output embeddings
        return F.normalize(embeddings, p=2, dim=1)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a training step.
        
        Args:
            batch: Tuple of (anchor, positive, negative) embeddings
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        anchor, positive, negative = batch
        
        # Project embeddings
        anchor_proj = self(anchor)
        positive_proj = self(positive)
        negative_proj = self(negative)
        
        # Calculate loss
        loss = self.triplet_loss(anchor_proj, positive_proj, negative_proj)
        
        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a validation step.
        
        Args:
            batch: Tuple of (anchor, positive, negative) embeddings and their labels
            batch_idx: Batch index
        """
        anchor, positive, negative, labels = batch
        
        # Project embeddings
        anchor_proj = self(anchor)
        positive_proj = self(positive)
        negative_proj = self(negative)
        
        # Calculate loss
        loss = self.triplet_loss(anchor_proj, positive_proj, negative_proj)
        
        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store embeddings for kNN evaluation (using anchors only)
        self.val_embeddings.append(anchor_proj.detach().cpu().numpy())
        # Use the labels directly
        self.val_labels.append(labels.detach().cpu().numpy())
    
    def on_validation_epoch_end(self) -> None:
        """Evaluate embeddings using kNN at the end of validation epoch."""
        # Concatenate all embeddings and labels
        all_embeddings = np.vstack(self.val_embeddings)
        all_labels = np.concatenate(self.val_labels)
        
        # Fit kNN classifier on 70% of validation data
        split_idx = int(0.7 * len(all_embeddings))
        X_train, y_train = all_embeddings[:split_idx], all_labels[:split_idx]
        X_test, y_test = all_embeddings[split_idx:], all_labels[split_idx:]
        
        # Fit and evaluate kNN
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Calculate metrics
        acc = (y_pred == y_test).mean()
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        # Log metrics
        self.log("val/knn_acc", acc, prog_bar=True)
        self.log("val/knn_balanced_acc", balanced_acc, prog_bar=True)
        
        # Reset storage
        self.val_embeddings = []
        self.val_labels = []
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a test step, similar to validation step."""
        anchor, positive, negative = batch
        
        # Project embeddings
        anchor_proj = self(anchor)
        positive_proj = self(positive)
        negative_proj = self(negative)
        
        # Store embeddings for final evaluation
        self.test_embeddings.append(anchor_proj.detach().cpu().numpy())
        
        # Get labels from the dataset for anchors
        dataset = self.trainer.test_dataloaders[0].dataset
        start_idx = batch_idx * self.trainer.test_dataloaders[0].batch_size
        end_idx = start_idx + len(anchor)
        anchor_indices = list(range(start_idx, end_idx))
        anchor_labels = [dataset.labels[i % len(dataset)] for i in anchor_indices]
        self.test_labels.append(np.array(anchor_labels))
    
    def on_test_epoch_end(self) -> None:
        """Evaluate embeddings using kNN at the end of test epoch."""
        # Concatenate all embeddings and labels
        all_embeddings = np.vstack(self.test_embeddings)
        all_labels = np.concatenate(self.test_labels)
        
        # Evaluate with kNN using cross-validation folds
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        # Use k-fold cross-validation for stable metrics
        n_folds = 5
        fold_size = len(all_embeddings) // n_folds
        metrics = {
            "acc": [],
            "balanced_acc": [],
            "f1": []
        }
        
        for i in range(n_folds):
            # Create train/test split
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else len(all_embeddings)
            
            X_test = all_embeddings[test_start:test_end]
            y_test = all_labels[test_start:test_end]
            
            X_train = np.vstack([all_embeddings[:test_start], all_embeddings[test_end:]])
            y_train = np.concatenate([all_labels[:test_start], all_labels[test_end:]])
            
            # Fit and evaluate
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            
            # Calculate metrics
            metrics["acc"].append((y_pred == y_test).mean())
            metrics["balanced_acc"].append(balanced_accuracy_score(y_test, y_pred))
            metrics["f1"].append(f1_score(y_test, y_pred, average='macro'))
        
        # Log metrics
        for name, values in metrics.items():
            self.log(f"test/{name}", np.mean(values), sync_dist=True)
        
        # Reset storage
        self.test_embeddings = []
        self.test_labels = []
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Use a simple ReduceLROnPlateau scheduler if provided
        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.lr_scheduler.get("mode", "min"),
                factor=self.hparams.lr_scheduler.get("factor", 0.5),
                patience=self.hparams.lr_scheduler.get("patience", 5),
                min_lr=self.hparams.lr_scheduler.get("min_lr", 1e-8)
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.lr_scheduler.get("monitor", "val/loss"),
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True
                }
            }
        
        return optimizer

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Project new proteins into the embedding space.
        
        Args:
            batch: Batch of protein embeddings
            batch_idx: Batch index
            
        Returns:
            Projected embeddings
        """
        # For single embeddings (not triplets)
        if not isinstance(batch, tuple):
            return self(batch)
        # For triplets, return anchor projection
        return self(batch[0])
