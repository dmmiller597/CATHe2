import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger
import torch
from pathlib import Path

from utils import get_logger, set_seed
from data.data_module import CATHeDataModule
from models.classifier import CATHeClassifier

log = get_logger()

def setup_callbacks(cfg: DictConfig) -> list:
    """Essential callbacks with dynamic metric handling"""
    return [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename='{epoch:02d}-{val_acc:.4f}',
            monitor=cfg.training.monitor_metric,
            mode=cfg.training.monitor_mode,
            save_top_k=cfg.training.save_top_k,
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

def setup_trainer(cfg: DictConfig, wandb_logger: WandbLogger) -> pl.Trainer:
    """Simplified trainer with automatic resource management"""
    return pl.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=cfg.training.max_epochs,
        callbacks=setup_callbacks(cfg),
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

@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Complete training workflow with automatic setup"""
    # Reproducibility
    set_seed(cfg.training.seed)

    # Data setup
    data_dir = Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir
    data_module = CATHeDataModule(
        data_dir=data_dir,
        **{k: v for k, v in cfg.data.items() if k != "data_dir"},
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )
    data_module.setup()
    
    # Model setup
    model = CATHeClassifier(
        **cfg.model,
        num_classes=data_module.num_classes
    )
    
    # Wandb Logger Setup
    wandb_logger = WandbLogger(
        project="CATHe",
        save_dir=cfg.training.log_dir,
        log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Training
    trainer = setup_trainer(cfg, wandb_logger)
    trainer.fit(model, data_module)
    
    # Final testing
    trainer.test(model, data_module, ckpt_path="best")
    
    # Finish Wandb run
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    main() 