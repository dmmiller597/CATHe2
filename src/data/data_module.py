import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple, List, Set
from pathlib import Path

from utils import get_logger

log = get_logger()

class CATHeDataset(Dataset):
    """Dataset class for CATH superfamily classification."""
    
    def __init__(self, embeddings_path: Path, labels_path: Path, valid_classes: Set[str] = None):
        """Initialize dataset with protein embeddings and their corresponding labels.
        
        Args:
            embeddings_path: Path to NPZ file containing ProtT5 embeddings
            labels_path: Path to CSV file containing SF labels
            valid_classes: Set of class names to include (if None, include all)
        """
        try:
            data = np.load(embeddings_path)
            labels_df = pd.read_csv(labels_path)
            
            # Filter by valid classes if specified
            if valid_classes is not None:
                mask = labels_df['SF'].isin(valid_classes)
                labels_df = labels_df[mask]
                
                # Check which key exists in the data and use the appropriate one
                if 'arr_0' in data:
                    self.embeddings = data['arr_0'][mask]
                else:
                    # We need to filter embeddings to match filtered labels
                    indices = np.where(mask)[0]
                    self.embeddings = data['embeddings'][indices]
            else:
                # No filtering, use all data
                if 'arr_0' in data:
                    self.embeddings = data['arr_0']
                    # Filter out problematic indices if they exist in this dataset
                    mask = ~np.isin(np.arange(len(self.embeddings)), [194048, 200243])
                    self.embeddings = self.embeddings[mask]
                    labels_df = labels_df[mask]
                else:
                    self.embeddings = data['embeddings']
            
            # Convert class names to integer codes
            self.class_names = labels_df['SF'].values
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
        min_samples_per_class: int = 10,  # New parameter for filtering
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
            min_samples_per_class: Minimum number of samples required per class
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
        self.min_samples_per_class = min_samples_per_class
        self.datasets: Dict[str, CATHeDataset] = {}
        self.valid_classes = None

    def balanced_class_sampler(self, labels: torch.Tensor) -> WeightedRandomSampler:
        """Create a weighted sampler that balances class representation.
        
        This implements inverse frequency weighting
        
        Args:
            labels: Tensor containing class labels
        
        Returns:
            WeightedRandomSampler for balanced training
        """
        # Calculate class counts
        class_counts = torch.bincount(labels).float()
        n_classes = len(class_counts)
        
        # Calculate inverse frequency weights (1/frequency)
        class_weights = 1.0 / (class_counts + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * n_classes
        
        # Log statistics
        min_weight = class_weights.min().item()
        max_weight = class_weights.max().item()
        weight_ratio = max_weight / (min_weight + 1e-10)
        
        log.info(f"Sampling weight statistics:")
        log.info(f"  - Min weight: {min_weight:.8f}")
        log.info(f"  - Max weight: {max_weight:.8f}")
        log.info(f"  - Max/Min ratio: {weight_ratio:.2f}")
        log.info(f"  - Unique classes: {n_classes}")
        
        # Map weights to samples
        sample_weights = class_weights[labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )

    def get_valid_classes(self) -> Set[str]:
        """Identify classes with sufficient samples in the training set."""
        # Read training labels
        train_labels_df = pd.read_csv(self.data_dir / self.train_labels)
        
        # Count occurrences of each class
        class_counts = train_labels_df['SF'].value_counts()
        
        # Filter classes with more than min_samples_per_class samples
        valid_classes = set(class_counts[class_counts > self.min_samples_per_class].index)
        
        log.info(f"Filtering classes with <= {self.min_samples_per_class} samples:")
        log.info(f"  - Original number of classes: {len(class_counts)}")
        log.info(f"  - Filtered number of classes: {len(valid_classes)}")
        log.info(f"  - Removed {len(class_counts) - len(valid_classes)} classes")
        
        return valid_classes

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage of training."""
        # Identify valid classes from training set (only needs to be done once)
        if self.valid_classes is None:
            self.valid_classes = self.get_valid_classes()
        
        if stage in (None, "fit"):
            # Create filtered datasets
            self.datasets["train"] = CATHeDataset(
                self.data_dir / self.train_embeddings,
                self.data_dir / self.train_labels,
                valid_classes=self.valid_classes
            )
            
            self.datasets["val"] = CATHeDataset(
                self.data_dir / self.val_embeddings,
                self.data_dir / self.val_labels,
                valid_classes=self.valid_classes
            )
            
            # Store number of classes for model configuration (after filtering)
            self.num_classes = len(pd.Categorical(self.datasets["train"].class_names).categories)
            
            # Create balanced class sampler for training
            self.train_sampler = self.balanced_class_sampler(
                self.datasets["train"].labels
            )
            
            # Log class distribution statistics after filtering
            class_counts = torch.bincount(self.datasets["train"].labels)
            log.info(f"Training set class statistics (after filtering):")
            log.info(f"  - Total classes: {len(class_counts)}")
            log.info(f"  - Min class size: {class_counts.min().item()}")
            log.info(f"  - Median class size: {torch.median(class_counts.float()).item():.1f}")
            log.info(f"  - Mean class size: {class_counts.float().mean().item():.1f}")
            log.info(f"  - Max class size: {class_counts.max().item()}")
            
        if stage == "test" and self.test_embeddings and self.test_labels:
            self.datasets["test"] = CATHeDataset(
                self.data_dir / self.test_embeddings,
                self.data_dir / self.test_labels,
                valid_classes=self.valid_classes
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