import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

from utils import get_logger

log = get_logger()

class CATHeDataset(Dataset):
    """Dataset class for CATH superfamily classification."""
    
    def __init__(self, embeddings_file: Path, allow_pickle: bool = True):
        """Initialize dataset with protein embeddings and their corresponding labels.
        
        Args:
            embeddings_file: Path to NPZ file containing ProtT5 embeddings and labels
            allow_pickle: Whether to allow pickle for loading the data
        """
        try:
            # Load both embeddings and labels from the same file
            data = np.load(embeddings_file, allow_pickle=True)  # Force allow_pickle=True
            self.embeddings = torch.from_numpy(data['embeddings']).float()
            
            # Handle string labels properly
            raw_labels = data['labels']
            self.label_encoder = pd.Categorical(raw_labels)
            self.labels = torch.tensor(self.label_encoder.codes, dtype=torch.long)
            
            # Store original string labels and indices for testing
            self.original_labels = raw_labels
            self.indices = np.arange(len(self.embeddings))
            
            log.info(f"Loaded dataset with {len(self.embeddings)} samples and {len(self.label_encoder.categories)} classes")
            log.info(f"Embedding dimension: {self.embeddings.shape[1]}")
            
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.embeddings[idx], self.labels[idx]

    def filter_by_mask(self, mask: List[bool]) -> None:
        """Filter dataset to keep only samples specified by mask.
        
        Args:
            mask: List of boolean values indicating which samples to keep
        """
        mask = torch.tensor(mask)
        self.embeddings = self.embeddings[mask]
        self.original_labels = self.original_labels[mask]
        
        # Re-encode labels to ensure consecutive indices
        self.label_encoder = pd.Categorical(self.original_labels)
        self.labels = torch.tensor(self.label_encoder.codes, dtype=torch.long)
        self.indices = np.arange(len(self.embeddings))
        
        log.info(f"Dataset filtered to {len(self)} samples with {len(self.label_encoder.categories)} classes")

class CATHeDataModule(pl.LightningDataModule):
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
        """Set up the datasets with memory-efficient filtering."""
        # Load training set categories first
        data = np.load(self.data_dir / self.train_embeddings, allow_pickle=True)
        train_categories = set(pd.Categorical(data['labels']).categories)
        log.info(f"Number of training classes: {len(train_categories)}")
        
        # Now load full training set
        self.datasets["train"] = CATHeDataset(
            embeddings_file=self.data_dir / self.train_embeddings,
            allow_pickle=True
        )
        
        # Load and filter validation set with pre-filtering
        data = np.load(self.data_dir / self.val_embeddings, allow_pickle=True)
        val_mask = np.array([label in train_categories for label in data['labels']])
        filtered_embeddings = data['embeddings'][val_mask]
        filtered_labels = data['labels'][val_mask]
        
        # Create temporary NPZ file for filtered data
        tmp_val_path = self.data_dir / 'tmp_val.npz'
        np.savez(tmp_val_path, embeddings=filtered_embeddings, labels=filtered_labels)
        self.datasets["val"] = CATHeDataset(embeddings_file=tmp_val_path)
        tmp_val_path.unlink()  # Clean up temporary file
        
        # Same for test set
        if self.test_embeddings:
            data = np.load(self.data_dir / self.test_embeddings, allow_pickle=True)
            test_mask = np.array([label in train_categories for label in data['labels']])
            filtered_embeddings = data['embeddings'][test_mask]
            filtered_labels = data['labels'][test_mask]
            
            tmp_test_path = self.data_dir / 'tmp_test.npz'
            np.savez(tmp_test_path, embeddings=filtered_embeddings, labels=filtered_labels)
            self.datasets["test"] = CATHeDataset(embeddings_file=tmp_test_path)
            tmp_test_path.unlink()  # Clean up temporary file
        
        # Store number of classes for model configuration
        self.num_classes = len(train_categories)

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            sampler=self._create_weighted_sampler(self.datasets["train"].labels),
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