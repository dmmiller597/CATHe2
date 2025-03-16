import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import random
from collections import defaultdict

from utils import get_logger

log = get_logger()

class TripletDataset(Dataset):
    """Dataset class for triplet-based contrastive learning of CATH superfamilies."""
    
    def __init__(self, embeddings_path: Path, labels_path: Path, mining_strategy: str = "random"):
        """Initialize dataset with protein embeddings and prepare for triplet generation.
        
        Args:
            embeddings_path: Path to NPZ file containing ProtT5 embeddings
            labels_path: Path to CSV file containing SF labels
            mining_strategy: Strategy for triplet selection ('random', 'semi-hard', 'hard')
        """
        try:
            data = np.load(embeddings_path)
            labels_df = pd.read_csv(labels_path)
            
            # Handle different embedding formats
            if 'arr_0' in data:
                self.embeddings = data['arr_0']
                log.info("Using arr_0 key format for embeddings")
            else:
                self.embeddings = data['embeddings']
                log.info("Using embeddings key format")
                
            # Get numerical codes for superfamily labels
            self.sf_labels = labels_df['SF'].values
            self.label_encoder = {sf: i for i, sf in enumerate(sorted(set(self.sf_labels)))}
            self.label_decoder = {i: sf for sf, i in self.label_encoder.items()}
            self.labels = np.array([self.label_encoder[sf] for sf in self.sf_labels])
            
            # Create index mapping for each class to enable efficient triplet generation
            self.label_to_indices = defaultdict(list)
            for idx, label in enumerate(self.labels):
                self.label_to_indices[label].append(idx)
                
            # Filter out classes with only a single example
            self.valid_classes = [label for label, indices in self.label_to_indices.items() 
                                 if len(indices) > 1]
            
            if len(self.valid_classes) < len(self.label_to_indices):
                log.warning(f"Removed {len(self.label_to_indices) - len(self.valid_classes)} classes with only a single example")
                
            # Set mining strategy
            self.mining_strategy = mining_strategy
            log.info(f"Using {mining_strategy} triplet mining strategy")
            
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise ValueError(f"Error loading embeddings from {embeddings_path}: {str(e)}")
    
    def __len__(self) -> int:
        """Return the number of potential anchor samples."""
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a triplet (anchor, positive, negative) from the dataset.
        
        Args:
            idx: Index of the anchor sample
            
        Returns:
            Tuple of (anchor, positive, negative) embeddings
        """
        anchor_embedding = self.embeddings[idx]
        anchor_label = self.labels[idx]
        
        # Find a positive sample (same class, different instance)
        pos_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        if not pos_indices:
            # Fallback if no other samples in this class
            pos_idx = idx
            log.warning(f"No positive samples found for class {self.label_decoder[anchor_label]}, using anchor as positive")
        else:
            pos_idx = random.choice(pos_indices)
        
        # Find a negative sample (different class)
        neg_label = random.choice([c for c in self.valid_classes if c != anchor_label])
        neg_idx = random.choice(self.label_to_indices[neg_label])
        
        # Get embeddings
        pos_embedding = self.embeddings[pos_idx]
        neg_embedding = self.embeddings[neg_idx]
        
        return (
            torch.tensor(anchor_embedding, dtype=torch.float),
            torch.tensor(pos_embedding, dtype=torch.float),
            torch.tensor(neg_embedding, dtype=torch.float)
        )


class ContrastiveDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for contrastive learning on CATH superfamilies."""
    
    def __init__(
        self,
        data_dir: str,
        train_embeddings: str,
        train_labels: str,
        val_embeddings: str,
        val_labels: str,
        test_embeddings: str = None,
        test_labels: str = None,
        batch_size: int = 128,
        num_workers: int = 4,
        mining_strategy: str = "random",
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
            mining_strategy: Strategy for triplet selection ('random', 'semi-hard', 'hard')
        """
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_labels = val_labels
        self.test_embeddings = test_embeddings
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mining_strategy = mining_strategy
        self.datasets = {}
        
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage of training."""
        if stage in (None, "fit"):
            self.datasets["train"] = TripletDataset(
                self.data_dir / self.train_embeddings,
                self.data_dir / self.train_labels,
                mining_strategy=self.mining_strategy
            )
            self.datasets["val"] = TripletDataset(
                self.data_dir / self.val_embeddings,
                self.data_dir / self.val_labels,
                mining_strategy=self.mining_strategy
            )
            # Store information about the embedding space
            self.embedding_dim = self.datasets["train"].embeddings.shape[1]
            self.num_classes = len(set(self.datasets["train"].labels))
            
            # Log statistics
            log.info(f"Training set statistics:")
            log.info(f"  - Total samples: {len(self.datasets['train'])}")
            log.info(f"  - Total classes: {self.num_classes}")
            log.info(f"  - Embedding dimension: {self.embedding_dim}")
            
        if stage == "test" and self.test_embeddings and self.test_labels:
            self.datasets["test"] = TripletDataset(
                self.data_dir / self.test_embeddings,
                self.data_dir / self.test_labels,
                mining_strategy=self.mining_strategy
            )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
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