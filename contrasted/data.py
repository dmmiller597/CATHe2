import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import lightning as L
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import get_logger

log = get_logger(__name__)

CATH_LEVEL_NAMES = {
    0: "Class",
    1: "Architecture",
    2: "Topology",
    3: "Homologous_Superfamily",
}


def get_level_label(sf: str, level: int) -> str:
    """Extract the CATH hierarchy label at the given level from an SF string."""
    parts = sf.split(".")
    return ".".join(parts[: level + 1]) if len(parts) > level else sf


class EmbeddingDataset(Dataset):
    """Dataset for precomputed embeddings, CATH labels, and sequence IDs."""

    def __init__(self, emb_path: Path, lbl_path: Path, level: int = 3):
        if level not in CATH_LEVEL_NAMES:
            raise ValueError(f"cath_level must be 0..3, got {level}")
        self.level = level

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


        sf_series = df["SF"].astype(str).apply(lambda s: get_level_label(s, level))
        unique_sf = sorted(sf_series.unique())
        encoder = {sf: idx for idx, sf in enumerate(unique_sf)}

        self.labels = torch.tensor(sf_series.map(encoder).to_numpy(), dtype=torch.long)
        self.label_decoder = {idx: sf for sf, idx in encoder.items()}

        self.num_classes = len(unique_sf)
        self.embedding_dim = self.embeddings.shape[1]

        if not (len(self.embeddings) == len(self.labels) == len(self.sequence_ids)):
            raise ValueError(
                "Mismatch between embeddings, labels, and sequence_ids lengths: "
                f"{len(self.embeddings)}, {len(self.labels)}, {len(self.sequence_ids)}"
            )
        log.info(f"Loaded {len(self)} samples for level {self.level}.")


    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Returns embedding, label index, and sequence ID."""
        emb = torch.from_numpy(self.embeddings[idx])
        lbl = self.labels[idx]
        seq_id = self.sequence_ids[idx]
        return emb, lbl, seq_id


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
        cath_level: int = 3,
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
            "cath_level", "batch_size", "num_workers", "pin_memory", "persistent_workers"
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
                ds = EmbeddingDataset(emb_path, lbl_path, self.hparams.cath_level)
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
        )

    def train_dataloader(self):
        if "train" not in self.datasets:
            self.setup("fit")
        return self._make_dataloader("train", True)

    def val_dataloader(self):
        if "val" not in self.datasets:
            self.setup("fit")
        return self._make_dataloader("val", False)

    def test_dataloader(self):
        if "test" not in self.datasets:
            self.setup("test")
        return self._make_dataloader("test", False)

    def get_label_decoder(self) -> Dict[int, str]:
        return self.datasets["train"].label_decoder

    def get_num_classes(self) -> int:
        return self.num_classes