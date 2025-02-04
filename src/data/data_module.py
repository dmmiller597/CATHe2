import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from utils import get_logger

log = get_logger()

class CATHeDataset(Dataset):
    """Dataset class for CATH superfamily classification."""
    
    def __init__(self, embeddings_path: Path, labels_path: Path):
        """Initialize dataset with protein embeddings and their corresponding labels.
        
        Args:
            embeddings_path: Path to NPZ file containing ProtT5 embeddings
            labels_path: Path to CSV file containing SF labels
        """
        try:
            # Load data into memory more efficiently
            self.embeddings = torch.from_numpy(np.load(embeddings_path)['arr_0']).float()
            labels_df = pd.read_csv(labels_path)
            codes = pd.Categorical(labels_df['SF']).codes
            # Use torch.tensor to force a copy and create a writable tensor
            self.labels = torch.tensor(codes, dtype=torch.long)
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (embedding, label) 
        """
        return self.embeddings[idx], self.labels[idx]

class CATHeDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for CATH superfamily classification."""
    
    def __init__(
        self,
        data_dir: str,
        train_embeddings: str,
        train_labels: str,
        val_embeddings: str,
        val_labels: str,
        test_embeddings: str = None,
        test_labels: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Initialize data module.
        
        Args:
            data_dir: Root directory containing data files
            train_embeddings: Path to training embeddings file
            train_labels: Path to training labels file
            val_embeddings: Path to validation embeddings file
            val_labels: Path to validation labels file
            test_embeddings: Path to test embeddings file (optional)
            test_labels: Path to test labels file (optional)
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
        """
        super().__init__()
        # Convert data_dir to absolute path if it's relative
        self.data_dir = Path(data_dir).resolve()
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_labels = val_labels
        self.test_embeddings = test_embeddings
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets: Dict[str, CATHeDataset] = {}

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage of training.
        
        Args:
            stage: Current stage ('fit' or 'test')
        """
        if stage in (None, "fit"):
            self.datasets["train"] = CATHeDataset(
                self.data_dir / self.train_embeddings,
                self.data_dir / self.train_labels
            )
            self.datasets["val"] = CATHeDataset(
                self.data_dir / self.val_embeddings,
                self.data_dir / self.val_labels
            )
            # Store number of classes for model configuration
            self.num_classes = len(pd.read_csv(self.data_dir / self.train_labels)['SF'].unique())
            
        if stage == "test" and self.test_embeddings and self.test_labels:
            self.datasets["test"] = CATHeDataset(
                self.data_dir / self.test_embeddings,
                self.data_dir / self.test_labels
            )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
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