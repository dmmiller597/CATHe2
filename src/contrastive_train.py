import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch
from pathlib import Path

from utils import get_logger, set_seed
from data.contrastive_data_module import ContrastiveDataModule
from models.contrasted import ContrastiveCATHeModel

log = get_logger()

@hydra.main(config_path="../config", config_name="contrastive", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main training function for contrastive learning model."""
    # Reproducibility
    set_seed(cfg.training.seed)
    
    # Set float32 matmul precision
    torch.set_float32_matmul_precision(cfg.training.accelerator.float32_matmul_precision)
    
    # Data setup
    data_dir = Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir
    data_module = ContrastiveDataModule(
        data_dir=data_dir,
        train_embeddings_file=cfg.data.train_embeddings,
        train_labels_file=cfg.data.train_labels,
        val_embeddings_file=cfg.data.val_embeddings,
        val_labels_file=cfg.data.val_labels,
        test_embeddings_file=cfg.data.test_embeddings if hasattr(cfg.data, "test_embeddings") else None,
        test_labels_file=cfg.data.test_labels if hasattr(cfg.data, "test_labels") else None,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )
    
    # Model setup
    model = ContrastiveCATHeModel(
        input_embedding_dim=cfg.data.embedding_dim,
        projection_hidden_dims=cfg.model.projection_dims,
        output_embedding_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        triplet_margin=cfg.model.margin,
        lr_scheduler_config=cfg.model.lr_scheduler,
        knn_neighbors=cfg.model.n_neighbors
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename='{epoch:02d}-{val/1nn_balanced_acc:.4f}',
            monitor=cfg.training.monitor_metric,
            mode=cfg.training.monitor_mode,
            save_last=True,
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor=cfg.training.monitor_metric,
            patience=cfg.training.early_stopping_patience,
            mode=cfg.training.monitor_mode
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar()
    ]
    
    # Wandb Logger
    wandb_logger = WandbLogger(
        project="CATHe-Contrastive",
        save_dir=cfg.training.log_dir,
        log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices=cfg.training.accelerator.devices,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=cfg.training.seed is not None,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=cfg.training.output_dir
    )
    
    # Training
    trainer.fit(model, data_module)
    
    # Final testing
    log.info("Starting final testing phase...")
    trainer.test(model, data_module, ckpt_path="best")
    log.info("Testing completed - check W&B dashboard for metrics")
    
    # Finish Wandb run
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    main() 