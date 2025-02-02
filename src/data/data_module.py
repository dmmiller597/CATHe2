import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, Any
from pathlib import Path

from utils import get_logger

log = get_logger()

class CATHeDataset(Dataset):
    """Dataset for CATH protein embeddings and labels."""
    
    def __init__(
        self,
        embeddings_path: str,
        labels_path: str,
        transform: Optional[callable] = None
    ):
        """Initialize dataset.
        
        Args:
            embeddings_path: Path to NPZ file containing embeddings
            labels_path: Path to CSV file containing labels
            transform: Optional transform to apply to embeddings
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If data dimensions mismatch
        """
        embeddings_path = Path(embeddings_path)
        labels_path = Path(labels_path)
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
        # Load data
        with np.load(embeddings_path) as data:
            self.embeddings = torch.FloatTensor(data['arr_0'])
            
        self.labels = torch.LongTensor(pd.read_csv(labels_path).values.squeeze())
        
        if len(self.embeddings) != len(self.labels):
            raise ValueError(
                f"Mismatch between embeddings ({len(self.embeddings)}) and "
                f"labels ({len(self.labels)})"
            )
            
        self.transform = transform
        log.info(f"Loaded dataset with {len(self)} samples")
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.embeddings)
        
    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (embedding, label)
        """
        embedding = self.embeddings[idx]
        if self.transform:
            embedding = self.transform(embedding)
        return embedding, self.labels[idx]

class CATHeDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for CATH classification."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train_embeddings: str,
        val_embeddings: str,
        test_embeddings: str,
        train_labels: str,
        val_labels: str,
        test_labels: str,
        num_workers: int = 4,
        transform: Optional[callable] = None
    ):
        """Initialize the data module.
        
        Args:
            data_dir: Root directory containing the data
            batch_size: Batch size for dataloaders
            train_embeddings: Path to training embeddings
            val_embeddings: Path to validation embeddings
            test_embeddings: Path to test embeddings
            train_labels: Path to training labels
            val_labels: Path to validation labels
            test_labels: Path to test labels
            num_workers: Number of workers for dataloaders (default: 4)
            transform: Optional transform to apply to embeddings
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
        self.file_paths = {
            'train': (Path(train_embeddings), Path(train_labels)),
            'val': (Path(val_embeddings), Path(val_labels)),
            'test': (Path(test_embeddings), Path(test_labels))
        }
        
        self.datasets = {split: None for split in ['train', 'val', 'test']}
        
    def prepare_data(self) -> None:
        """Verify all data files exist."""
        for split, (emb_path, label_path) in self.file_paths.items():
            if not emb_path.exists():
                raise FileNotFoundError(f"{split} embeddings not found at: {emb_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"{split} labels not found at: {label_path}")
                
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training/validation/testing.
        
        Args:
            stage: Stage to setup ('fit' or 'test')
        """
        if stage == "fit" or stage is None:
            for split in ['train', 'val']:
                self.datasets[split] = CATHeDataset(
                    str(self.file_paths[split][0]),
                    str(self.file_paths[split][1]),
                    transform=self.transform
                )
                log.info(f"Set up {split} dataset")
            
        if stage == "test" or stage is None:
            self.datasets['test'] = CATHeDataset(
                str(self.file_paths['test'][0]),
                str(self.file_paths['test'][1]),
                transform=self.transform
            )
            log.info("Set up test dataset")
            
    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        return DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self) -> DataLoader:
        """Create the test data loader."""
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 