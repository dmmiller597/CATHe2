# file: contrasted/eval.py

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from data import ContrastiveDataModule
from model import ContrastiveCATHeModel
from utils import configure_logging, set_seed, get_logger

log = get_logger(__name__)

@hydra.main(config_path="../config", config_name="contrastive", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # ---- setup logging & seed ----
    configure_logging()
    set_seed(cfg.training.seed)
    torch.set_float32_matmul_precision(
        getattr(cfg.training, "float32_matmul_precision", "medium")
    )

    # ---- resolve checkpoint & level ----
    ckpt_path: str = cfg.eval.ckpt_path
    if not Path(ckpt_path).is_file():
        log.error(f"Checkpoint not found: {ckpt_path}")
        return
    cath_level: int = cfg.eval.get("cath_level", cfg.training.cath_levels[-1])
    log.info(f"Evaluating ckpt={ckpt_path} @ CATH level {cath_level}")

    # ---- build data module ----
    root = Path(hydra.utils.get_original_cwd())
    dm = ContrastiveDataModule(
        data_dir=str(root / cfg.data.data_dir),
        train_embeddings_file=cfg.data.train_embeddings,
        train_labels_file=cfg.data.train_labels,
        val_embeddings_file=cfg.data.val_embeddings,
        val_labels_file=cfg.data.val_labels,
        test_embeddings_file=cfg.data.get("test_embeddings", None),
        test_labels_file=cfg.data.get("test_labels", None),
        cath_level=cath_level,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    dm.setup(stage="fit")   # loads train + val (needed to infer dims)
    dm.setup(stage="test")  # loads test

    # ---- load model ----
    model = ContrastiveCATHeModel.load_from_checkpoint(
        ckpt_path,
        strict=False,   # allow minor hparam changes
    )

    # ---- test-only trainer ----
    trainer = L.Trainer(
        accelerator="auto",
        devices=cfg.training.accelerator.devices,
        precision=cfg.training.precision,
        logger=False,             # no new logger needed
        enable_progress_bar=True, # optional
    )
    trainer.test(model=model, datamodule=dm)  # runs test_step + on_test_epoch_end()

if __name__ == "__main__":
    main()