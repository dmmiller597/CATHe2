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

def create_run_dir(base_output_dir: Path) -> Path:
    """
    Create a single directory for the training run and return its path.
    """
    run_dir = base_output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_datamodule(cfg: DictConfig) -> ContrastiveDataModule:
    """
    Instantiate and setup the data module for CATH Homologous Superfamily.
    """
    data_path = Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir
    dm = ContrastiveDataModule(
        data_dir=str(data_path),
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


def build_model(cfg: DictConfig, output_dir: Path) -> ContrastiveCATHeModel:
    """
    Instantiate a new model.
    """
    vis_args = {
        "tsne_viz_dir": str(output_dir),
        "umap_viz_dir": str(output_dir),
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


def build_callbacks(cfg: DictConfig, output_dir: Path) -> list:
    """
    Create callbacks: checkpointing, early stopping, LR monitor, progress bar.
    """
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="checkpoint-{epoch:02d}-{val_centroid_f1_macro:.4f}",
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


def build_logger(cfg: DictConfig) -> WandbLogger:
    """
    Configure and return a WandbLogger for this training run.
    """
    return WandbLogger(
        project="CATHe-Contrastive",
        name=None,  # Let wandb generate a unique name
        log_model=False,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(config_path="../config", config_name="contrastive", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main entry for contrastive learning on the CATH Homologous Superfamily level.
    """
    configure_logging()
    set_seed(cfg.training.seed)
    torch.set_float32_matmul_precision(cfg.training.accelerator.float32_matmul_precision)

    log.info("=== Training Contrastive Model ===")

    logger = build_logger(cfg)

    # Create directories for outputs
    base_output_dir = Path(hydra.utils.get_original_cwd()) / cfg.training.output_dir
    run_dir = create_run_dir(base_output_dir)
    log.info(f"Run output directory: {run_dir}")

    # Setup data, model, and callbacks
    dm = build_datamodule(cfg)
    model = build_model(cfg, run_dir)
    callbacks = build_callbacks(cfg, run_dir)

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
        default_root_dir=str(run_dir),
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