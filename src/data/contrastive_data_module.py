import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from collections import defaultdict
import logging

# Use standard logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EmbeddingDataset(Dataset):
    """
    Dataset class that loads pre-computed embeddings and corresponding labels.

    This dataset simply returns an embedding and its integer label for a given index.
    It does not perform any triplet mining itself.
    """

    def __init__(self, embeddings_path: Path, labels_path: Path):
        """
        Initializes the dataset by loading embeddings and labels.

        Args:
            embeddings_path: Path to the NPZ file containing embeddings (key: 'embeddings').
            labels_path: Path to the CSV file containing labels (column: 'SF').

        Raises:
            FileNotFoundError: If embedding or label files do not exist.
            ValueError: If required keys/columns are missing or data loading fails.
        """
        log.info(f"Initializing EmbeddingDataset from {embeddings_path} and {labels_path}")
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Embedding file not found: {embeddings_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        try:
            data = np.load(embeddings_path)
            if 'embeddings' not in data:
                raise ValueError(f"'embeddings' key not found in NPZ file: {embeddings_path}")
            self.embeddings = data['embeddings'].astype(np.float32)

            labels_df = pd.read_csv(labels_path)
            if 'SF' not in labels_df.columns:
                raise ValueError(f"'SF' column not found in CSV file: {labels_path}")

            # Store string labels and create integer encodings
            self.sf_labels = labels_df['SF'].values
            unique_sorted_labels = sorted(list(set(self.sf_labels)))
            self.label_encoder = {sf: i for i, sf in enumerate(unique_sorted_labels)}
            self.label_decoder = {i: sf for sf, i in self.label_encoder.items()}
            self.labels = np.array([self.label_encoder[sf] for sf in self.sf_labels], dtype=np.int64)

            self.num_classes = len(self.label_encoder)
            self.embedding_dim = self.embeddings.shape[1]

            log.info(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embedding_dim} and {self.num_classes} classes.")

            # Sanity check: embeddings and labels count should match
            if len(self.embeddings) != len(self.labels):
                raise ValueError(
                    f"Mismatch between number of embeddings ({len(self.embeddings)}) "
                    f"and labels ({len(self.labels)})."
                )

        except Exception as e:
            log.error(f"Error initializing EmbeddingDataset: {e}", exc_info=True)
            raise ValueError(f"Failed to load data: {str(e)}") from e

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the embedding and its integer label at the given index.
        """
        embedding = torch.from_numpy(self.embeddings[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label


class ContrastiveDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for contrastive learning.

    Handles the creation of training, validation, and testing datasets and dataloaders
    using the EmbeddingDataset. It assumes embeddings are pre-computed.
    """

    def __init__(
        self,
        data_dir: str,
        train_embeddings_file: str,
        train_labels_file: str,
        val_embeddings_file: str,
        val_labels_file: str,
        test_embeddings_file: Optional[str] = None,
        test_labels_file: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initializes the DataModule.

        Args:
            data_dir: Root directory containing data files.
            train_embeddings_file: Filename of training embeddings NPZ file (relative to data_dir).
            train_labels_file: Filename of training labels CSV file (relative to data_dir).
            val_embeddings_file: Filename of validation embeddings NPZ file (relative to data_dir).
            val_labels_file: Filename of validation labels CSV file (relative to data_dir).
            test_embeddings_file: Optional filename of test embeddings NPZ file.
            test_labels_file: Optional filename of test labels CSV file.
            batch_size: Number of samples per batch.
            num_workers: Number of subprocesses for data loading.
            pin_memory: If True, DataLoader will copy Tensors into CUDA pinned memory before returning them.
            persistent_workers: If True, workers will not be shut down after an epoch.
        """
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.train_embeddings_path = self.data_dir / train_embeddings_file
        self.train_labels_path = self.data_dir / train_labels_file
        self.val_embeddings_path = self.data_dir / val_embeddings_file
        self.val_labels_path = self.data_dir / val_labels_file

        self.test_embeddings_path = self.data_dir / test_embeddings_file if test_embeddings_file else None
        self.test_labels_path = self.data_dir / test_labels_file if test_labels_file else None

        # Store hyperparameters
        self.save_hyperparameters(
            "batch_size", "num_workers", "pin_memory", "persistent_workers"
        )

        self.datasets: Dict[str, EmbeddingDataset] = {}
        self.embedding_dim: Optional[int] = None
        self.num_classes: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        """
        Loads and prepares datasets for the specified stage ('fit', 'test', or None).

        Args:
            stage: The stage for which to set up data ('fit', 'test', or None for both).
        """
        log.info(f"Setting up data module for stage: {stage}")

        # Setup for training and validation ('fit' stage)
        if stage in (None, "fit"):
            if "train" not in self.datasets:
                log.info("Loading training data...")
                self.datasets["train"] = EmbeddingDataset(
                    self.train_embeddings_path, self.train_labels_path
                )
                self.embedding_dim = self.datasets["train"].embedding_dim
                self.num_classes = self.datasets["train"].num_classes
            
            if "val" not in self.datasets:
                log.info("Loading validation data...")
                self.datasets["val"] = EmbeddingDataset(
                    self.val_embeddings_path, self.val_labels_path
                )
                # Ensure validation data compatibility
                if self.datasets["val"].embedding_dim != self.embedding_dim:
                     log.warning(f"Validation embedding dim ({self.datasets['val'].embedding_dim}) "
                                 f"differs from train ({self.embedding_dim})!")
                log.info("Validation data loaded.")

        # Setup for testing ('test' stage)
        if stage in (None, "test"):
            if self.test_embeddings_path and self.test_labels_path:
                if "test" not in self.datasets:
                    log.info("Loading test data...")
                    self.datasets["test"] = EmbeddingDataset(
                        self.test_embeddings_path, self.test_labels_path
                    )
                    # Ensure test data compatibility
                    if self.datasets["test"].embedding_dim != self.embedding_dim:
                         log.warning(f"Test embedding dim ({self.datasets["test"].embedding_dim}) "
                                     f"differs from train ({self.embedding_dim})!")
                    log.info("Test data loaded.")
            elif stage == "test":
                log.warning("Test stage requested but test data paths were not provided.")

    def _get_dataloader(self, split: str, shuffle: bool) -> Optional[DataLoader]:
        """Helper function to create a DataLoader for a specific split."""
        if split not in self.datasets:
            log.warning(f"Dataset for split '{split}' not found.")
            return None

        is_train = (split == "train")
        persist = self.hparams.persistent_workers if is_train else False
        prefetch = 2 if is_train else None

        try:
            # Use balanced class sampling for training
            if is_train:
                # Get all labels
                labels = self.datasets[split].labels
                
                # Create class-balanced batch sampler
                batch_sampler = BalancedClassSampler(
                    labels=labels,
                    samples_per_class=2,  # 2 samples per class
                    classes_per_batch=min(1024, self.datasets[split].num_classes)  # Adjust based on GPU memory
                )
                
                loader = DataLoader(
                    self.datasets[split],
                    batch_sampler=batch_sampler,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    persistent_workers=persist and self.hparams.num_workers > 0,
                    prefetch_factor=prefetch if self.hparams.num_workers > 0 else None,
                )
            else:
                # Standard dataloader for validation/test
                loader = DataLoader(
                    self.datasets[split],
                    batch_size=self.hparams.batch_size,
                    shuffle=shuffle,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    persistent_workers=persist and self.hparams.num_workers > 0,
                    prefetch_factor=prefetch if self.hparams.num_workers > 0 else None,
                    drop_last=is_train
                )
            return loader
        except Exception as e:
            log.error(f"Failed to create DataLoader for split '{split}': {e}", exc_info=True)
            return None

    def train_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the training set."""
        return self._get_dataloader("train", shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the validation set."""
        return self._get_dataloader("val", shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the test set."""
        return self._get_dataloader("test", shuffle=False)

    def get_label_decoder(self) -> Optional[Dict[int, str]]:
        """Returns the label decoder dictionary (int -> string label)."""
        if "train" in self.datasets:
            return self.datasets["train"].label_decoder
        log.warning("Label decoder requested but training dataset is not loaded.")
        return None

class BalancedClassSampler(BatchSampler):
    """Samples N instances for each class in the dataset."""
    
    def __init__(self, labels, samples_per_class=2, classes_per_batch=None):
        self.labels_list = labels
        self.samples_per_class = samples_per_class
        self.class_indices = self._build_class_indices()
        self.classes_per_batch = classes_per_batch or len(self.class_indices)
        # Limit classes per batch if needed
        self.classes_per_batch = min(self.classes_per_batch, len(self.class_indices))
        
    def _build_class_indices(self):
        indices = defaultdict(list)
        for i, label in enumerate(self.labels_list):
            indices[label].append(i)
        return indices
    
    def __iter__(self):
        for _ in range(len(self)):
            # Select which classes to include in this batch
            class_keys = list(self.class_indices.keys())
            # If we can't fit all classes in a batch, randomly select some
            if len(class_keys) > self.classes_per_batch:
                class_keys = np.random.choice(class_keys, self.classes_per_batch, replace=False)
                
            batch_indices = []
            
            # For each selected class
            for class_idx in class_keys:
                class_samples = self.class_indices[class_idx]
                # Handle cases where a class has fewer than requested samples
                samples_to_take = min(self.samples_per_class, len(class_samples))
                
                # Select samples without replacement if possible
                if len(class_samples) <= samples_to_take:
                    # If we have fewer samples than needed, take all
                    batch_indices.extend(class_samples)
                else:
                    # Otherwise, randomly sample without replacement
                    batch_indices.extend(
                        np.random.choice(class_samples, samples_to_take, replace=False)
                    )
            
            np.random.shuffle(batch_indices)  # Shuffle to prevent class ordering bias
            yield batch_indices
    
    def __len__(self):
        # Depends on your desired number of batches per epoch
        # One approach: each class should be seen roughly same number of times
        max_samples_per_class = max(len(samples) for samples in self.class_indices.values())
        return max(1, max_samples_per_class // self.samples_per_class)