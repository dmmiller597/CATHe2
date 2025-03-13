import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import pyarrow.parquet as pq

from utils import get_logger

log = get_logger()

class CATHeDataset(Dataset):
    """Dataset class for CATH superfamily classification using NPZ files."""
    
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
                print("using arr_0 key for the original CATHe dataset)")
            else:
                self.embeddings = data['embeddings']
                print("using embeddings key for TED dataset")
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


class ParquetCATHeDataset(Dataset):
    """Dataset class for CATH superfamily classification using Parquet files."""
    
    def __init__(self, parquet_path: Path):
        """Initialize dataset with protein embeddings and their corresponding labels from a Parquet file.
        
        Args:
            parquet_path: Path to Parquet file containing embeddings and SF labels
        """
        try:
            # Load parquet file
            self.df = pd.read_parquet(parquet_path)
            
            # Convert embeddings from list of arrays to numpy array
            self.embeddings = np.stack(self.df['embedding'].values)
            
            # Convert SF labels to categorical codes
            codes = pd.Categorical(self.df['SF']).codes
            self.labels = torch.tensor(codes, dtype=torch.long)
            
            log.info(f"Loaded {len(self.df)} samples from {parquet_path}")
            log.info(f"Embeddings shape: {self.embeddings.shape}")
            log.info(f"Number of unique classes: {len(self.df['SF'].unique())}")
            
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise ValueError(f"Error loading data from {parquet_path}: {str(e)}")
        
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
        train_labels: str = None,
        val_embeddings: str = None,
        val_labels: str = None,
        test_embeddings: str = None,
        test_labels: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sampling_beta: float = 0.9999,
        use_parquet: bool = False,
    ):
        """Initialize data module.
        
        Args:
            data_dir: Root directory containing data files
            train_embeddings: Path to training embeddings file (NPZ) or Parquet file
            train_labels: Path to training labels file (CSV) - not needed if use_parquet=True
            val_embeddings: Path to validation embeddings file (NPZ) or Parquet file
            val_labels: Path to validation labels file (CSV) - not needed if use_parquet=True
            test_embeddings: Path to test embeddings file (NPZ or Parquet) (optional)
            test_labels: Path to test labels file (CSV) - not needed if use_parquet=True (optional)
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            sampling_beta: Smoothing parameter in [0,1]; 0 = no reweighting, 1 = inverse frequency
            use_parquet: Whether to use Parquet files that contain both embeddings and labels
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
        self.use_parquet = use_parquet
        self.datasets: Dict[str, Union[CATHeDataset, ParquetCATHeDataset]] = {}

    def balanced_class_sampler(self, labels, beta):
        print(f"Using sampling_beta: {beta}")
        class_counts = torch.bincount(labels)
        print(f"Class counts: {class_counts}")
        # Log min/max/mean of calculated weights
        weights = (1.0 - beta) + beta * (1.0 / class_counts[labels])
        print(f"Weight stats: min={weights.min()}, max={weights.max()}, mean={weights.mean()}")
        return WeightedRandomSampler(weights, len(weights))

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage of training."""
        if stage in (None, "fit"):
            # Load training data
            if self.use_parquet:
                self.datasets["train"] = ParquetCATHeDataset(
                    self.data_dir / self.train_embeddings
                )
                self.datasets["val"] = ParquetCATHeDataset(
                    self.data_dir / self.val_embeddings
                )
                # Get unique SF values from the Parquet file for model configuration
                train_df = pd.read_parquet(self.data_dir / self.train_embeddings, columns=["SF"])
                self.num_classes = len(train_df['SF'].unique())
            else:
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
            
        if stage == "test" and self.test_embeddings:
            if self.use_parquet:
                self.datasets["test"] = ParquetCATHeDataset(
                    self.data_dir / self.test_embeddings
                )
            else:
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