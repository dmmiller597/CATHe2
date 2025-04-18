import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning as L
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from utils import get_logger

log = get_logger()

class CATHeDataset(Dataset):
    """Dataset class for CATH superfamily classification."""
    
    def __init__(self, embeddings_path: Path, labels_path: Path, label_encoder: LabelEncoder = None):
        """Initialize dataset with protein embeddings and their corresponding labels.
        
        Args:
            embeddings_path: Path to NPZ file containing ProtT5 embeddings
            labels_path: Path to CSV file containing SF labels
            label_encoder: LabelEncoder for consistent label indexing
        """
        try:
            data = np.load(embeddings_path)
            labels_df = pd.read_csv(labels_path)
            # Check which key exists in the data and use the appropriate one
            if 'arr_0' in data:
                self.embeddings = data['arr_0']
                print("using arr_0 key for the original CATHe dataset)")
            else:
                self.embeddings = data['embeddings']
                print("using embeddings key for TED dataset")
            if label_encoder is not None:
                codes = label_encoder.transform(labels_df['SF'])
            else:
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

class CATHeDataModule(L.LightningDataModule):
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
        # Infer number of classes and initialize train sampler immediately
        train_labels_path = self.data_dir / self.train_labels
        try:
            df = pd.read_csv(train_labels_path)
            # Fit a single LabelEncoder on training labels for consistent mapping
            self.label_encoder = LabelEncoder().fit(df['SF'])
            codes = self.label_encoder.transform(df['SF'])
            self.num_classes = len(self.label_encoder.classes_)
            # compute inverse-frequency weights
            counts = np.bincount(codes)
            weights = (1.0 - self.sampling_beta) + self.sampling_beta * (1.0 / counts[codes])
            sampler_weights = torch.tensor(weights, dtype=torch.double)
            self.train_sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
            log.info(f"Initialized {self.num_classes} classes and sampler from {train_labels_path}")
        except Exception as e:
            log.warning(f"Failed to infer num_classes or sampler: {e}")
            self.num_classes = None
            self.train_sampler = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage of training."""
        if stage in (None, "fit"):
            self.datasets["train"] = CATHeDataset(
                self.data_dir / self.train_embeddings,
                self.data_dir / self.train_labels,
                label_encoder=self.label_encoder
            )
            self.datasets["val"] = CATHeDataset(
                self.data_dir / self.val_embeddings,
                self.data_dir / self.val_labels,
                label_encoder=self.label_encoder
            )
            # num_classes and train_sampler are already initialized in __init__
            
        if stage == "test" and self.test_embeddings and self.test_labels:
            self.datasets["test"] = CATHeDataset(
                self.data_dir / self.test_embeddings,
                self.data_dir / self.test_labels,
                label_encoder=self.label_encoder
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