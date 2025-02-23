import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from utils import get_logger

log = get_logger()

class CATHeTEDDataset(Dataset):
    """Dataset class for CATH superfamily classification."""
    
    def __init__(self, embeddings_path: Path):
        """Initialize dataset with protein embeddings and their corresponding labels.
        
        Args:
            embeddings_path: Path to NPZ file containing ProtT5 embeddings and labels
        """
        try:
            # Load both embeddings and labels from the same file
            data = np.load(embeddings_path)
            embeddings = data['embeddings']
            labels = data['labels']
            
            self.embeddings = torch.from_numpy(embeddings).float()
            # Convert string labels to categorical codes
            self.label_encoder = pd.Categorical(labels)
            self.labels = torch.tensor(self.label_encoder.codes, dtype=torch.long)
            
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.embeddings[idx], self.labels[idx]

class CATHeTEDDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for CATH superfamily classification."""
    
    def __init__(
        self,
        data_dir: str,
        train_embeddings: str,
        val_embeddings: str,
        test_embeddings: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Initialize data module.
        
        Args:
            data_dir: Root directory containing data files
            train_embeddings: Path to training embeddings file
            val_embeddings: Path to validation embeddings file
            test_embeddings: Path to test embeddings file (optional)
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
        """
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.train_embeddings = train_embeddings
        self.val_embeddings = val_embeddings
        self.test_embeddings = test_embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets: Dict[str, CATHeDataset] = {}

    def _create_weighted_sampler(self, labels: torch.Tensor) -> WeightedRandomSampler:
        """Create a weighted sampler to handle class imbalance."""
        try:
            # Calculate class weights
            class_counts = torch.bincount(labels).float()
            # Add small epsilon to prevent division by zero
            class_weights = 1.0 / (class_counts + 1e-8)
            # Normalize weights to sum to 1
            class_weights = class_weights / class_weights.sum()
            # Map weights to samples
            sample_weights = class_weights[labels]
            
            return WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(labels),
                replacement=True
            )
        except Exception as e:
            log.error(f"Error creating weighted sampler: {e}")
            raise

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage of training."""
        if stage in (None, "fit"):
            self.datasets["train"] = CATHeDataset(
                self.data_dir / self.train_embeddings
            )
            self.datasets["val"] = CATHeDataset(
                self.data_dir / self.val_embeddings
            )
            # Store number of classes for model configuration
            self.num_classes = len(self.datasets["train"].label_encoder.categories)
            
            # Create weighted sampler for training
            self.train_sampler = self._create_weighted_sampler(self.datasets["train"].labels)
            
        if stage == "test" and self.test_embeddings:
            self.datasets["test"] = CATHeDataset(
                self.data_dir / self.test_embeddings
            )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test data loader if test data is available."""
        if "test" in self.datasets:
            return DataLoader(
                self.datasets["test"],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None 