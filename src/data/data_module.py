import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
            data = np.load(embeddings_path)
            labels_df = pd.read_csv(labels_path)
            # Check which key exists in the data and use the appropriate one
            if 'arr_0' in data:
                self.embeddings = data['arr_0']
                # Filter out problematic indices if they exist in this dataset
                mask = ~np.isin(np.arange(len(self.embeddings)), [194048, 200243])
                self.embeddings = self.embeddings[mask]
                labels_df = labels_df[mask]
            else:
                self.embeddings = data['embeddings']
            
            codes = pd.Categorical(labels_df['SF']).codes
            self.labels = torch.tensor(codes, dtype=torch.long)
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise ValueError(f"Error loading embeddings from {embeddings_path}: {str(e)}")
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
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
        sampling_beta: float = 0.9999,
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
            sampling_beta: Smoothing parameter in [0,1]; 0 = no reweighting, 1 = inverse frequency
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
        self.sampling_beta = sampling_beta
        self.datasets: Dict[str, CATHeDataset] = {}

    def balanced_class_sampler(self, labels: torch.Tensor, beta: float = 0.9999) -> WeightedRandomSampler:
        """Create a weighted sampler that balances class representation using effective numbers.
        
        This implements a more flexible weighting strategy based on "Effective Number of Samples"
        (Cui et al., 2019) with a controllable beta parameter.
        
        Args:
            labels: Tensor containing class labels
            beta: Smoothing parameter in [0,1]; 0 = no reweighting, 1 = inverse frequency
        
        Returns:
            WeightedRandomSampler for balanced training
        """
        # Calculate class counts
        class_counts = torch.bincount(labels).float()
        n_classes = len(class_counts)
        n_samples = len(labels)
        
        # Calculate effective number weights: (1-beta)/(1-beta^n)
        # When beta=1, this becomes exactly inverse weighting
        # When beta=0, all classes get the same weight
        effective_numbers = (1.0 - torch.pow(beta, class_counts)) / (1.0 - beta + 1e-8)
        class_weights = 1.0 / (effective_numbers + 1e-8)
        
        # Normalize weights so they sum to n_classes (like before)
        class_weights = class_weights / class_weights.sum() * n_classes
        
        # Log statistics
        min_weight = class_weights.min().item()
        max_weight = class_weights.max().item()
        weight_ratio = max_weight / (min_weight + 1e-10)
        
        log.info(f"Effective number sampling (beta={beta:.4f}):")
        log.info(f"  - Min weight: {min_weight:.8f}")
        log.info(f"  - Max weight: {max_weight:.8f}")
        log.info(f"  - Max/Min ratio: {weight_ratio:.2f}")
        log.info(f"  - Unique classes: {n_classes}")
        
        # Map weights to samples
        sample_weights = class_weights[labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=n_samples,
            replacement=True
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage of training."""
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
            
            # Create balanced class sampler for training
            self.train_sampler = self.balanced_class_sampler(
                self.datasets["train"].labels,
                beta=self.sampling_beta
            )
            
            # Log class distribution statistics
            class_counts = torch.bincount(self.datasets["train"].labels)
            log.info(f"Training set class statistics:")
            log.info(f"  - Total classes: {len(class_counts)}")
            log.info(f"  - Min class size: {class_counts.min().item()}")
            log.info(f"  - Median class size: {torch.median(class_counts.float()).item():.1f}")
            log.info(f"  - Mean class size: {class_counts.float().mean().item():.1f}")
            log.info(f"  - Max class size: {class_counts.max().item()}")
            
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