import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import lightning as L
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import get_logger, convert_sf_string_to_list

log = get_logger(__name__)

def get_superfamily_label(sf: str) -> str:
    """Extract the CATH Homologous Superfamily label (level 3) from an SF string."""
    parts = sf.split(".")
    return ".".join(parts[:4]) if len(parts) > 3 else sf


class EmbeddingDataset(Dataset):
    """
    Dataset for precomputed embeddings, CATH Homologous Superfamily labels,
    and sequence IDs.
    """

    def __init__(self, emb_path: Path, lbl_path: Path):
        if not emb_path.is_file():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if not lbl_path.is_file():
            raise FileNotFoundError(f"Labels file not found: {lbl_path}")

        data = np.load(emb_path)
        if "embeddings" not in data:
            raise KeyError(f"'embeddings' key not found in {emb_path}")
        self.embeddings = data["embeddings"].astype(np.float32)

        df = pd.read_csv(lbl_path)
        if "SF" not in df.columns:
            raise KeyError(f"'SF' column not found in {lbl_path}")
        if "sequence_id" not in df.columns:
            log.warning(f"'sequence_id' column not found in {lbl_path}, cannot retrieve sequence IDs.")
            self.sequence_ids = ["unknown"] * len(df) # Placeholder
        else:
             self.sequence_ids = df["sequence_id"].astype(str).to_list()

        # For OverlapLoss: hierarchical list of strings
        self.hierarchical_labels = (
            df["SF"].astype(str).apply(convert_sf_string_to_list).to_list()
        )

        # For SupConLoss and metrics: integer labels
        sf_series = df["SF"].astype(str).apply(get_superfamily_label)
        unique_sf = sorted(sf_series.unique())
        encoder = {sf: idx for idx, sf in enumerate(unique_sf)}

        self.labels = torch.tensor(sf_series.map(encoder).to_numpy(), dtype=torch.long)
        self.label_decoder = {idx: sf for sf, idx in encoder.items()}

        self.num_classes = len(unique_sf)
        self.embedding_dim = self.embeddings.shape[1]

        if not (len(self.embeddings) == len(self.labels) == len(self.sequence_ids) == len(self.hierarchical_labels)):
            raise ValueError(
                "Mismatch between embeddings, labels, sequence_ids, and hierarchical_labels lengths: "
                f"{len(self.embeddings)}, {len(self.labels)}, {len(self.sequence_ids)}, {len(self.hierarchical_labels)}"
            )
        log.info(f"Loaded {len(self)} samples for Homologous Superfamily level.")


    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, List[str]], str]:
        """Returns embedding, (integer label, hierarchical label), and sequence ID."""
        emb = torch.from_numpy(self.embeddings[idx])
        lbl = self.labels[idx]
        hier_lbl = self.hierarchical_labels[idx]
        seq_id = self.sequence_ids[idx]
        return emb, (lbl, hier_lbl), seq_id


class ContrastiveDataModule(L.LightningDataModule):
    """LightningDataModule for CATH embeddings with weighted sampling."""

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
        super().__init__()
        base = Path(data_dir).resolve()
        self.paths: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {
            "train": (base / train_embeddings_file, base / train_labels_file),
            "val":   (base / val_embeddings_file,   base / val_labels_file),
            "test":  (
                base / test_embeddings_file if test_embeddings_file else None,
                base / test_labels_file if test_labels_file else None,
            ),
        }
        self.save_hyperparameters(
            "batch_size", "num_workers", "pin_memory", "persistent_workers"
        )

        self.datasets: Dict[str, EmbeddingDataset] = {}
        self.embedding_dim: Optional[int] = None
        self.num_classes: Optional[int] = None
        self._train_weights: Optional[torch.Tensor] = None

    def prepare_data(self) -> None:
        """Ensure all required data files exist."""
        missing = [
            p
            for paths in self.paths.values()
            for p in paths
            if p and not p.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing data files: {missing}")
        log.info("All required data files found.")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load datasets for the specified stage ('fit' for train/val, 'test' for test)."""
        stage_map = {"train": "fit", "val": "fit", "test": "test"}
        for split, req_stage in stage_map.items():
            emb_path, lbl_path = self.paths[split]
            if emb_path and lbl_path and (stage is None or stage == req_stage):
                ds = EmbeddingDataset(emb_path, lbl_path)
                self.datasets[split] = ds
                if split == "train":
                    self.embedding_dim = ds.embedding_dim
                    self.num_classes = ds.num_classes
                    self._train_weights = self._compute_weights(ds.labels)
                else:
                    if ds.embedding_dim != self.embedding_dim:
                        log.warning(
                            f"{split} embedding_dim ({ds.embedding_dim}) "
                            f"!= train ({self.embedding_dim})"
                        )

    def _compute_weights(self, labels: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(labels, minlength=self.num_classes)
        weights = (1.0 / (counts.float() + 1e-6))[labels]
        return weights.double()

    def _make_dataloader(self, split: str, train: bool):
        ds = self.datasets.get(split)
        if ds is None:
            return None
        sampler = (
            WeightedRandomSampler(self._train_weights, len(ds), replacement=True)
            if train
            else None
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if train else False,
            drop_last=train,
            collate_fn=collate_embedding_batch,
        )

    def train_dataloader(self):
        return self._make_dataloader("train", True)

    def val_dataloader(self):
        return self._make_dataloader("val", False)

    def test_dataloader(self):
        return self._make_dataloader("test", False)

    def get_label_decoder(self) -> Dict[int, str]:
        if "train" not in self.datasets:
            self.setup("fit") # Ensure dataset is loaded for decoder access
        return self.datasets["train"].label_decoder

    def get_num_classes(self) -> int:
        if "train" not in self.datasets:
            self.setup("fit") # Ensure dataset is loaded for num_classes access
        return self.num_classes

# --- Custom collate function to keep hierarchical labels per-sample ---

def collate_embedding_batch(batch):
    """Custom collate_fn for `EmbeddingDataset`.

    Keeps the hierarchical label list as *per-sample* structure instead of
    letting PyTorch default collation transpose it. The returned structure
    matches what `ContrastiveCATHeModel.training_step` expects:

        embeddings:   Tensor [B, D]
        ((int_labels: Tensor[B]), hier_labels: List[List[str]]),
        sequence_ids: List[str]
    """
    embeddings = torch.stack([item[0] for item in batch])           # (B, D)
    int_labels = torch.stack([item[1][0] for item in batch])        # (B,)
    hier_labels = [item[1][1] for item in batch]                    # List[B][str]
    sequence_ids = [item[2] for item in batch]                      # List[B]
    return embeddings, (int_labels, hier_labels), sequence_ids
