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
            # Use contrastive sampling for training
            if is_train:
                # Get all labels
                labels = self.datasets[split].labels
                
                # Create contrastive batch sampler
                batch_sampler = ContrastiveBatchSampler(
                    labels=labels,
                    batch_size=self.hparams.batch_size,
                    samples_per_class=2,  # 2 samples per class for contrastive pairs
                    preselect_size=18
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

    def preview_batch(self, split: str = "train"):
        """
        Prints the first batch labels and a preview of the embeddings.
        
        Args:
            split: The dataset split to preview ("train", "val", or "test")
        
        Returns:
            Tuple of (embeddings, labels) from the first batch or None if data not available
        """
        if split not in self.datasets:
            log.warning(f"Dataset for split '{split}' not found. Call setup() first.")
            return None
        
        dataloader = self._get_dataloader(split, shuffle=False)
        if not dataloader:
            log.warning(f"Could not create dataloader for '{split}' split.")
            return None
            
        # Get the first batch
        for batch in dataloader:
            embeddings, labels = batch
            break
        
        # Print label information
        label_decoder = self.get_label_decoder()
        decoded_labels = [label_decoder[l.item()] for l in labels] if label_decoder else labels.tolist()
        
        print(f"\n{'='*50}")
        print(f"PREVIEW OF FIRST {split.upper()} BATCH:")
        print(f"{'='*50}")
        
        print(f"\nBatch size: {len(labels)}")
        print(f"Label counts:")
        label_counts = defaultdict(int)
        for label in labels.tolist():
            label_counts[label] += 1
        
        for label_id, count in sorted(label_counts.items()):
            label_name = label_decoder[label_id] if label_decoder else f"Class {label_id}"
            print(f"  {label_name}: {count} samples")
        
        # Print embedding information
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")
        print(f"Embedding stats:")
        print(f"  Mean: {embeddings.mean().item():.4f}")
        print(f"  Std:  {embeddings.std().item():.4f}")
        print(f"  Min:  {embeddings.min().item():.4f}")
        print(f"  Max:  {embeddings.max().item():.4f}")
        
        # Print a few sample embeddings (first 3 dimensions of first 5 samples)
        print("\nSample embeddings (first 3 dimensions of first 5 samples):")
        for i in range(min(5, len(embeddings))):
            print(f"  Sample {i} (label={decoded_labels[i]}): {embeddings[i, :3].tolist()}")
        
        print(f"\n{'='*50}\n")
        
        return embeddings, labels

class ContrastiveBatchSampler(BatchSampler):
    """
    Memory-efficient batch sampler for contrastive learning with class-balanced sampling.
    
    Pre-selects a fixed number of indices per class and samples from this pool to create batches.
    """
    
    def __init__(self, labels, batch_size=1024, samples_per_class=3, preselect_size=18):
        """
        Initialize the contrastive batch sampler.
        
        Args:
            labels: Array-like of integer class labels for each sample
            batch_size: Target batch size
            samples_per_class: Minimum number of samples to include from each selected class
            preselect_size: Number of samples to pre-select per class
        """
        self.labels = np.asarray(labels)
        self.num_samples = len(labels)
        self.samples_per_class = samples_per_class
        self.preselect_size = preselect_size
        
        # Get unique classes and their frequencies
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.unique_classes = unique_labels
        self.num_classes = len(unique_labels)
        
        # Store class frequencies for sampling weights
        self.class_counts = {label: count for label, count in zip(unique_labels, counts)}
        
        # Calculate effective batch size and classes per batch
        self.classes_per_batch = max(1, batch_size // samples_per_class)
        self.classes_per_batch = min(self.classes_per_batch, self.num_classes)
        self.effective_batch_size = self.classes_per_batch * samples_per_class
        
        # Calculate class sampling weights (inverse square root of frequency)
        weights = 1.0 / np.sqrt(counts)
        self.class_weights = weights / weights.sum()
        
        # Determine batches per epoch
        self.num_batches = self._calculate_num_batches()
        
        # Pre-compute class indices (major speedup)
        self.class_indices = {}
        for label in self.unique_classes:
            self.class_indices[label] = np.where(self.labels == label)[0]
            
        # Pre-select indices for each class
        self.preselected_indices = {}
        for label in self.unique_classes:
            indices = self.class_indices[label]
            if len(indices) <= self.preselect_size:
                # Use all available samples if fewer than preselect_size
                self.preselected_indices[label] = indices.copy()
            else:
                # Randomly select preselect_size samples
                self.preselected_indices[label] = np.random.choice(
                    indices, size=self.preselect_size, replace=False
                )
    
    def _calculate_num_batches(self):
        """Calculate optimal number of batches per epoch."""
        # Base on dataset size and ensure all classes are represented
        coverage_factor = min(2.0, max(1.0, np.log10(self.num_classes)))
        samples_covered = self.num_samples * 0.7  # Aim to cover 70% of dataset each epoch
        
        # Scale by dataset complexity
        if self.num_classes > 100:
            # For many classes, we don't need to see every sample in each epoch
            samples_covered *= (100 / self.num_classes) ** 0.5
        
        return max(100, int(samples_covered / self.effective_batch_size))
    
    def __iter__(self):
        """Generate balanced batches using pre-selected indices."""
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Select classes for this batch using weighted sampling
            batch_classes = np.random.choice(
                self.unique_classes,
                size=self.classes_per_batch,
                replace=False,
                p=self.class_weights
            )
            
            # For each selected class, sample instances from pre-selected indices
            for cls in batch_classes:
                cls_indices = self.preselected_indices[cls]
                
                # Handle class size appropriately
                if len(cls_indices) <= self.samples_per_class:
                    # Use all available samples and repeat if necessary
                    samples = list(cls_indices)
                    if len(samples) < self.samples_per_class:
                        additional = np.random.choice(
                            cls_indices, 
                            size=self.samples_per_class - len(samples),
                            replace=True
                        ).tolist()
                        samples.extend(additional)
                else:
                    # Randomly sample from pre-selected indices
                    samples = np.random.choice(
                        cls_indices,
                        size=self.samples_per_class,
                        replace=False
                    ).tolist()
                
                batch_indices.extend(samples)
            
            # Shuffle indices to avoid class-based ordering
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return self.num_batches