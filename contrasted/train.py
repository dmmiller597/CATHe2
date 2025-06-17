import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
import torch
from pathlib import Path

from utils import get_logger, set_seed, configure_logging
from data import ContrastiveDataModule
from model import ContrastiveCATHeModel

log = get_logger(__name__)

def create_output_dirs(base_output_dir: Path, run_name: str) -> dict[str, Path]:
    """
    Create directories for the training run and return their paths.
    """
    run_dir = base_output_dir / run_name
    subdirs = {
        "root": "",
        "ckpt": "checkpoints",
        "tsne": "tsne_plots",
        "umap": "umap_plots",
    }
    dirs = {key: (run_dir / sub if sub else run_dir) for key, sub in subdirs.items()}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_datamodule(cfg: DictConfig) -> ContrastiveDataModule:
    """
    Instantiate and setup the data module for CATH Homologous Superfamily.
    """
    dm = ContrastiveDataModule(
        data_dir=Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir,
        train_embeddings_file=cfg.data.train_embeddings,
        train_labels_file=cfg.data.train_labels,
        val_embeddings_file=cfg.data.val_embeddings,
        val_labels_file=cfg.data.val_labels,
        test_embeddings_file=getattr(cfg.data, "test_embeddings", None),
        test_labels_file=getattr(cfg.data, "test_labels", None),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    dm.setup(stage="fit")
    return dm


def build_model(cfg: DictConfig, dirs: dict[str, Path]) -> ContrastiveCATHeModel:
    """
    Instantiate a new model.
    """
    vis_args = {
        "tsne_viz_dir": str(dirs["tsne"]),
        "umap_viz_dir": str(dirs["umap"]),
    }

    log.info("Initializing new model.")
    return ContrastiveCATHeModel(
        input_embedding_dim=cfg.data.embedding_dim,
        projection_hidden_dims=cfg.model.projection_dims,
        output_embedding_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        visualization_method=cfg.model.visualization_method,
        enable_visualization=cfg.model.enable_visualization,
        seed=cfg.training.seed,
        **vis_args,
    )


def build_callbacks(cfg: DictConfig, ckpt_dir: Path, run_name: str) -> list:
    """
    Create callbacks: checkpointing, early stopping, LR monitor, progress bar.
    """
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}-{{epoch:02d}}-{{val/centroid_f1_macro:.4f}}",
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


def build_logger(cfg: DictConfig, run_name: str) -> WandbLogger:
    """
    Configure and return a WandbLogger for this training run.
    """
    wandb_logger = WandbLogger(
        project="CATHe-Contrastive",
        name=run_name,
        log_model=False,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return wandb_logger


@hydra.main(config_path="../config", config_name="contrastive", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main entry for contrastive learning on the CATH Homologous Superfamily level.
    """
    configure_logging()
    set_seed(cfg.training.seed)
    torch.set_float32_matmul_precision(cfg.training.accelerator.float32_matmul_precision)

    run_name = "contrastive_superfamily"
    log.info(f"=== Training CATH Homologous Superfamily ===")

    base_output_dir = Path(hydra.utils.get_original_cwd()) / cfg.training.output_dir
    dirs = create_output_dirs(base_output_dir, run_name)
    log.info(f"Run output directory: {dirs['root']}")

    dm = build_datamodule(cfg)
    model = build_model(cfg, dirs)
    callbacks = build_callbacks(cfg, dirs["ckpt"], run_name)
    logger = build_logger(cfg, run_name)

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

    checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path

    if not best_path or not Path(best_path).exists():
        log.error("Training finished, but no checkpoint was saved.")
        if logger:
            logger.experiment.finish(exit_code=1)
        return

    log.info(f"Best checkpoint saved to: {best_path}")

    if dm.test_dataloader() is not None:
        log.info("--- Starting Test Phase ---")
        trainer.test(model=model, datamodule=dm, ckpt_path=best_path)

    if logger:
        logger.experiment.finish()

    log.info("Training and testing finished.")


if __name__ == "__main__":
    main() 