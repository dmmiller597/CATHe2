import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
import torch
from pathlib import Path

from utils import get_logger, set_seed, configure_logging
from data import ContrastiveDataModule, CATH_LEVEL_NAMES
from model import ContrastiveCATHeModel

log = get_logger(__name__)

def create_level_dirs(base_output_dir: Path, cath_level: int, level_suffix: str) -> dict[str, Path]:
    """
    Create directories for a given CATH level and return their paths.
    """
    # define the root directory for this level
    level_dir = base_output_dir / f"level_{cath_level}_{level_suffix}"
    # mapping of directory keys to subfolder names (empty for root)
    subdirs = {
        "root": "",
        "ckpt": "checkpoints",
        "tsne": "tsne_plots",
        "umap": "umap_plots",
    }
    # build full paths using a dict comprehension
    dirs = {key: (level_dir / sub if sub else level_dir) for key, sub in subdirs.items()}
    # create all directories
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_datamodule(cfg: DictConfig, cath_level: int) -> ContrastiveDataModule:
    """
    Instantiate and setup the data module for the given CATH level.
    """
    dm = ContrastiveDataModule(
        data_dir=Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir,
        train_embeddings_file=cfg.data.train_embeddings,
        train_labels_file=cfg.data.train_labels,
        val_embeddings_file=cfg.data.val_embeddings,
        val_labels_file=cfg.data.val_labels,
        test_embeddings_file=getattr(cfg.data, "test_embeddings", None),
        test_labels_file=getattr(cfg.data, "test_labels", None),
        cath_level=cath_level,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    dm.setup(stage="fit")
    return dm


def build_model(cfg: DictConfig, dirs: dict[str, Path], last_ckpt: str | None) -> ContrastiveCATHeModel:
    """
    Instantiate a new model or load from the last checkpoint.
    """
    vis_args = {
        "tsne_viz_dir": str(dirs["tsne"]),
        "umap_viz_dir": str(dirs["umap"]),
    }

    if last_ckpt is None:
        log.info("Initializing new model.")
        return ContrastiveCATHeModel(
            input_embedding_dim=cfg.data.embedding_dim,
            projection_hidden_dims=cfg.model.projection_dims,
            output_embedding_dim=cfg.model.output_dim,
            dropout=cfg.model.dropout,
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            warmup_epochs=cfg.model.warmup_epochs,
            warmup_start_factor=cfg.model.warmup_start_factor,
            visualization_method=cfg.model.visualization_method,
            enable_visualization=cfg.model.enable_visualization,
            seed=cfg.training.seed,
            **vis_args,
        )
    else:
        log.info(f"Loading model from checkpoint: {last_ckpt}")
        return ContrastiveCATHeModel.load_from_checkpoint(
            last_ckpt,
            strict=False,
            learning_rate=cfg.model.learning_rate,
            seed=cfg.training.seed,
            enable_visualization=cfg.model.enable_visualization,
            **vis_args,
        )


def build_callbacks(cfg: DictConfig, ckpt_dir: Path, level_suffix: str) -> list:
    """
    Create callbacks: checkpointing, early stopping, LR monitor, progress bar.
    """
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{level_suffix}-{{epoch:02d}}-{{val/knn_balanced_acc:.4f}}",
        monitor=cfg.training.monitor_metric,
        mode=cfg.training.monitor_mode,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor=cfg.training.monitor_metric,
        patience=cfg.training.early_stopping_patience,
        mode=cfg.training.monitor_mode,
        verbose=True,
        check_on_train_epoch_end=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = RichProgressBar()
    return [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]


def build_logger(cfg: DictConfig, log_dir: Path, cath_level: int, level_suffix: str) -> WandbLogger:
    """
    Configure and return a WandbLogger for this training level.
    """
    wandb_logger = WandbLogger(
        project="CATHe-Contrastive-Hierarchical",
        name=level_suffix,
        log_model=False,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_logger.experiment.config.update({
        "current_cath_level": cath_level,
        "current_cath_level_name": level_suffix,
    })
    return wandb_logger


def train_one_level(
    cfg: DictConfig,
    base_output_dir: Path,
    cath_level: int,
    last_ckpt: str | None
) -> str | None:
    """
    Run training and optional testing for one hierarchical level.
    Returns the best checkpoint path or None if training failed.
    """
    level_suffix = CATH_LEVEL_NAMES.get(cath_level, f"Level_{cath_level}")
    dirs = create_level_dirs(base_output_dir, cath_level, level_suffix)

    dm = build_datamodule(cfg, cath_level)
    model = build_model(cfg, dirs, last_ckpt)
    callbacks = build_callbacks(cfg, dirs["ckpt"], level_suffix)
    logger = build_logger(cfg, dirs["root"], cath_level, level_suffix)

    trainer = L.Trainer(
        accelerator="auto",
        devices=cfg.training.accelerator.devices,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=str(dirs["root"]),
        num_sanity_val_steps=2,
    )

    trainer.fit(model, datamodule=dm)

    checkpoint_cb = callbacks[0]
    best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    if not best_path or not Path(best_path).exists():
        log.warning(f"No checkpoint saved for level {level_suffix}.")
        return None
    log.info(f"Best checkpoint for level {level_suffix}: {best_path}")

    if dm.test_dataloader() is not None:
        trainer.test(model=model, datamodule=dm, ckpt_path=best_path)
    logger.experiment.finish()
    return best_path

@hydra.main(config_path="../config", config_name="contrastive", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main entry: hierarchical contrastive learning across CATH levels.
    """
    configure_logging()
    set_seed(cfg.training.seed)
    torch.set_float32_matmul_precision(cfg.training.accelerator.float32_matmul_precision)

    base_output_dir = Path(cfg.training.output_dir)
    log.info(f"Base output directory: {base_output_dir}")

    last_ckpt: str | None = None
    for cath_level in cfg.training.cath_levels:
        log.info(f"=== Training CATH Level {cath_level} ===")
        last_ckpt = train_one_level(cfg, base_output_dir, cath_level, last_ckpt)
        if last_ckpt is None:
            log.error("Stopping early due to missing checkpoint.")
            break

    log.info("Hierarchical training finished for all specified CATH levels.")

if __name__ == "__main__":
    main() 