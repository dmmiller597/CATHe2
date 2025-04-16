import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from collections import defaultdict
import logging
from utils import get_logger
import warnings # Import warnings

# Use standard logging
log = get_logger(__name__)
# logging.basicConfig(level=logging.INFO) # Avoid double config if root logger is configured

# Define CATH level names for clarity
CATH_LEVEL_NAMES = {
    0: "Class",
    1: "Architecture",
    2: "Topology",
    3: "Homologous Superfamily"
}

class EmbeddingDataset(Dataset):
    """
    Dataset class that loads pre-computed embeddings and corresponding CATH labels
    for a specific hierarchy level.
    """

    def __init__(self, embeddings_path: Path, labels_path: Path, cath_level: int = 3):
        """
        Initializes the dataset by loading embeddings and labels for a specific CATH level.

        Args:
            embeddings_path: Path to the NPZ file containing embeddings (key: 'embeddings').
            labels_path: Path to the CSV file containing labels (column: 'SF').
            cath_level: The level of the CATH hierarchy to use for labels (0=C, 1=A, 2=T, 3=H).

        Raises:
            FileNotFoundError: If embedding or label files do not exist.
            ValueError: If required keys/columns are missing, cath_level is invalid,
                        or data loading fails.
        """
        level_name = CATH_LEVEL_NAMES.get(cath_level, f"Level {cath_level}")
        log.info(f"Initializing EmbeddingDataset for CATH {level_name} ({cath_level}) from {embeddings_path} and {labels_path}")

        if not (0 <= cath_level <= 3):
            raise ValueError(f"Invalid cath_level: {cath_level}. Must be between 0 and 3.")
        self.cath_level = cath_level

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

            # Extract the specified level from C.A.T.H hierarchy
            # Example: SF = '1.10.630.10'
            # Level 0: '1'
            # Level 1: '1.10'
            # Level 2: '1.10.630'
            # Level 3: '1.10.630.10'
            def get_level_label(sf_str: str, level: int) -> str:
                parts = sf_str.split('.')
                if len(parts) > level:
                    return ".".join(parts[:level + 1])
                else:
                    # Handle cases where the SF string doesn't go deep enough (use the deepest available)
                    # Or assign a special 'unknown' label if preferred
                    # warnings.warn(f"SF string '{sf_str}' too short for level {level}. Using full string.")
                    return sf_str # Use the full string as the label in this case

            self.sf_labels = [get_level_label(sf, self.cath_level) for sf in labels_df['SF'].astype(str)]

            unique_sorted_labels = sorted(list(set(self.sf_labels)))
            self.label_encoder = {sf: i for i, sf in enumerate(unique_sorted_labels)}
            self.label_decoder = {i: sf for sf, i in self.label_encoder.items()}
            self.labels = np.array([self.label_encoder[sf] for sf in self.sf_labels], dtype=np.int64)

            self.num_classes = len(self.label_encoder)
            self.embedding_dim = self.embeddings.shape[1]

            log.info(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embedding_dim}.")
            log.info(f"Found {self.num_classes} unique labels for CATH {level_name}: {unique_sorted_labels[:10]}...") # Log only first few

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
        Return the embedding and its integer label at the given index for the specified CATH level.
        """
        embedding = torch.from_numpy(self.embeddings[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label


class CATHeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CATH embedding datasets, configurable for CATH hierarchy level.
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
        cath_level: int = 3, # Add cath_level parameter
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initializes the CATHeDataModule for a specific CATH level.

        Args:
            data_dir: Root directory containing data files.
            train_embeddings_file: Filename of training embeddings NPZ (relative to data_dir).
            train_labels_file: Filename of training labels CSV (relative to data_dir).
            val_embeddings_file: Filename of validation embeddings NPZ (relative to data_dir).
            val_labels_file: Filename of validation labels CSV (relative to data_dir).
            test_embeddings_file: Optional filename of test embeddings NPZ.
            test_labels_file: Optional filename of test labels CSV.
            cath_level: The level of the CATH hierarchy to use (0=C, 1=A, 2=T, 3=H).
            batch_size: Number of samples per batch.
            num_workers: Number of subprocesses for data loading.
            pin_memory: If True, DataLoader copies Tensors into CUDA pinned memory.
            persistent_workers: If True, workers are not shut down after an epoch.
        """
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.train_embeddings_path = self.data_dir / train_embeddings_file
        self.train_labels_path = self.data_dir / train_labels_file
        self.val_embeddings_path = self.data_dir / val_embeddings_file
        self.val_labels_path = self.data_dir / val_labels_file

        self.test_embeddings_path = self.data_dir / test_embeddings_file if test_embeddings_file else None
        self.test_labels_path = self.data_dir / test_labels_file if test_labels_file else None

        # Store hyperparameters using save_hyperparameters()
        # Include cath_level here
        self.save_hyperparameters(
            "batch_size", "num_workers", "pin_memory", "persistent_workers", "cath_level"
        )

        self.datasets: Dict[str, EmbeddingDataset] = {}
        self.embedding_dim: Optional[int] = None
        self.num_classes: Optional[int] = None
        self._train_sampler_weights: Optional[torch.Tensor] = None # Cache sampler weights

    def prepare_data(self):
        """Download data if needed. Pytorch Lightning ensures this runs only on 1 GPU."""
        # Data is assumed pre-downloaded. Check file existence.
        paths_to_check = [
            self.train_embeddings_path, self.train_labels_path,
            self.val_embeddings_path, self.val_labels_path
        ]
        if self.test_embeddings_path: paths_to_check.append(self.test_embeddings_path)
        if self.test_labels_path: paths_to_check.append(self.test_labels_path)

        missing_files = [p for p in paths_to_check if not p.is_file()]
        if missing_files:
            log.error(f"Missing data files: {missing_files}")
            raise FileNotFoundError(f"Missing required data files: {missing_files}")
        log.info("All required data files found.")


    def setup(self, stage: Optional[str] = None):
        """
        Loads datasets for the specified stage ('fit', 'test', 'predict'), using the configured cath_level.

        Args:
            stage: The stage for which to set up data.
        """
        level_name = CATH_LEVEL_NAMES.get(self.hparams.cath_level, f"Level {self.hparams.cath_level}")
        log.info(f"Setting up data module for stage: {stage}, CATH Level: {level_name} ({self.hparams.cath_level})")

        # Clear previous datasets and sampler weights if setup is called again for a different level/stage
        self.datasets = {}
        self._train_sampler_weights = None
        self.num_classes = None
        self.embedding_dim = None


        # Setup for training and validation ('fit' stage)
        if stage == "fit" or stage is None:
            # Always reload train for 'fit' stage based on current cath_level
            log.info("Loading training data...")
            self.datasets["train"] = EmbeddingDataset(
                self.train_embeddings_path, self.train_labels_path, self.hparams.cath_level
            )
            # Infer dimensions from training data
            self.embedding_dim = self.datasets["train"].embedding_dim
            self.num_classes = self.datasets["train"].num_classes
            log.info(f"Train dataset loaded. Embedding dim: {self.embedding_dim}, Num classes for level {self.hparams.cath_level}: {self.num_classes}")

            # Precompute sampler weights
            labels = self.datasets["train"].labels
            class_counts = np.bincount(labels, minlength=self.num_classes)
            # Add small epsilon to avoid division by zero if a class has zero samples
            class_weights = 1.0 / (class_counts + 1e-6)
            self._train_sampler_weights = torch.from_numpy(class_weights[labels]).double()

            # Always reload val for 'fit' stage
            log.info("Loading validation data...")
            self.datasets["val"] = EmbeddingDataset(
                self.val_embeddings_path, self.val_labels_path, self.hparams.cath_level
            )
            if self.datasets["val"].embedding_dim != self.embedding_dim:
                    log.warning(f"Validation embedding dim ({self.datasets['val'].embedding_dim}) "
                                f"differs from train ({self.embedding_dim})!")
            # Note: Validation label space might differ from train, which is okay.
            log.info(f"Validation data loaded. Num classes for level {self.hparams.cath_level}: {self.datasets['val'].num_classes}")

        # Setup for testing ('test' stage)
        if stage == "test" or stage is None:
            if self.test_embeddings_path and self.test_labels_path:
                 # Always reload test for 'test' stage
                log.info("Loading test data...")
                self.datasets["test"] = EmbeddingDataset(
                    self.test_embeddings_path, self.test_labels_path, self.hparams.cath_level
                )
                if self.embedding_dim is None and "train" in self.datasets: # If only testing, infer from train if possible
                    self.embedding_dim = self.datasets["train"].embedding_dim
                elif self.embedding_dim is None: # If only testing and no train, infer from test
                     self.embedding_dim = self.datasets["test"].embedding_dim

                if self.datasets["test"].embedding_dim != self.embedding_dim:
                    log.warning(f"Test embedding dim ({self.datasets['test'].embedding_dim}) "
                                f"differs from inferred dim ({self.embedding_dim})!")
                log.info(f"Test data loaded. Num classes for level {self.hparams.cath_level}: {self.datasets['test'].num_classes}")
            elif stage == "test":
                 log.warning("Test stage requested but test data paths were not provided in config.")


    def _get_dataloader(self, split: str) -> Optional[DataLoader]:
        """Helper function to create a DataLoader for a specific split."""
        if split not in self.datasets:
            log.error(f"Dataset for split '{split}' not found. Setup may be needed.")
            if split == 'test' and not (self.test_embeddings_path and self.test_labels_path):
                 log.warning(f"Test dataset not loaded (likely missing files). Cannot create test dataloader.")
                 return None
            return None

        dataset = self.datasets[split]
        is_train = (split == "train")

        # Define parameters based on the split
        sampler = None
        shuffle = False # Default: No shuffle (for val/test)
        drop_last = False # Default: Keep last batch (for val/test)
        # Disable persistent workers by default, enable only for train if configured
        persistent_workers = False

        if is_train:
            # --- Training Setup ---
            if self._train_sampler_weights is None:
                log.error("Training sampler weights not computed. Call setup('fit') first.")
                # Attempt recovery (optional, could just return None)
                try:
                    log.warning("Attempting to compute sampler weights on the fly...")
                    labels = self.datasets["train"].labels
                    num_classes = self.datasets["train"].num_classes
                    class_counts = np.bincount(labels, minlength=num_classes)
                    class_weights = 1.0 / (class_counts + 1e-6)
                    self._train_sampler_weights = torch.from_numpy(class_weights[labels]).double()
                except Exception as e:
                     log.error(f"Failed to compute sampler weights on the fly: {e}")
                     return None # Cannot proceed without weights

            sampler = WeightedRandomSampler(
                weights=self._train_sampler_weights,
                num_samples=len(dataset),
                replacement=True
            )
            shuffle = False  # Sampler handles shuffling, so DataLoader shuffle must be False
            drop_last = True # Drop last incomplete batch for training consistency
            persistent_workers = (self.hparams.persistent_workers and self.hparams.num_workers > 0)
        # else: For val/test, the defaults (sampler=None, shuffle=False, drop_last=False, persistent_workers=False) are used.

        # Create the DataLoader
        try:
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                shuffle=shuffle,             # Explicitly False for val/test
                sampler=sampler,             # Explicitly None for val/test
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=persistent_workers,
                drop_last=drop_last,
            )
            return loader
        except Exception as e:
            log.error(f"Failed to create DataLoader for split '{split}': {e}", exc_info=True)
            return None

    def train_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the training set with weighted random sampling."""
        # Ensure setup was called if datasets are missing
        if "train" not in self.datasets:
            log.warning("Train dataset not loaded. Calling setup('fit').")
            self.setup('fit')
        return self._get_dataloader("train")

    def val_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the validation set."""
        if "val" not in self.datasets:
             log.warning("Validation dataset not loaded. Calling setup('fit').")
             self.setup('fit')
        return self._get_dataloader("val")

    def test_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the test set."""
        if "test" not in self.datasets:
             log.warning("Test dataset not loaded. Calling setup('test').")
             self.setup('test')
        return self._get_dataloader("test")

    def get_label_decoder(self) -> Optional[Dict[int, str]]:
        """Returns the label decoder dictionary (int -> string CATH label for the current level)."""
        # Try accessing from train, val, or test dataset if available
        for split in ["train", "val", "test"]:
            if split in self.datasets and hasattr(self.datasets[split], 'label_decoder'):
                return self.datasets[split].label_decoder
        log.warning("Label decoder requested but no datasets seem loaded or have a decoder.")
        # Attempt to load train dataset to get decoder if not available
        if "train" not in self.datasets:
            try:
                self.setup('fit')
                if "train" in self.datasets:
                    return self.datasets["train"].label_decoder
            except Exception as e:
                log.error(f"Failed to setup datasets to get label decoder: {e}")
        return None

    def get_num_classes(self) -> Optional[int]:
         """Returns the number of classes for the current level."""
         if self.num_classes is not None:
             return self.num_classes
         # Infer from datasets if not set
         for split in ["train", "val", "test"]:
             if split in self.datasets and hasattr(self.datasets[split], 'num_classes'):
                 self.num_classes = self.datasets[split].num_classes
                 return self.num_classes
         log.warning("Number of classes requested but no datasets seem loaded.")
         # Attempt to load train dataset to get num_classes if not available
         if "train" not in self.datasets:
             try:
                 self.setup('fit')
                 if "train" in self.datasets:
                     self.num_classes = self.datasets["train"].num_classes
                     return self.num_classes
             except Exception as e:
                 log.error(f"Failed to setup datasets to get num_classes: {e}")
         return None