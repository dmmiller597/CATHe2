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
import wandb
import os
from omegaconf import open_dict

from utils import get_logger, set_seed
from data.data_module import CATHeDataModule
from models.classifier import CATHeClassifier

log = get_logger()

def setup_callbacks(cfg: DictConfig) -> list:
    """Essential callbacks with dynamic metric handling"""
    return [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename='{epoch:02d}-{val/balanced_acc:.4f}',
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

def setup_trainer(cfg: DictConfig, wandb_logger: WandbLogger) -> pl.Trainer:
    """Simplified trainer with automatic resource management"""
    return pl.Trainer(
        accelerator='auto',
        devices=cfg.training.accelerator.devices,
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

@hydra.main(config_path="../config", config_name="ted_s30", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Complete training workflow with automatic setup"""
    
    # Check if this is a sweep run
    if "wandb" in cfg and "sweep" in cfg.wandb:
        run_sweep_agent(cfg)
    else:
        # Your existing training code
        run_single_training(cfg)

def run_sweep_agent(cfg: DictConfig):
    """Run as a W&B sweep agent"""
    # Initialize the sweep agent
    def sweep_train():
        # Get a new config from W&B sweep
        with wandb.init() as run:
            # Copy the sweep configuration into the Hydra config
            sweep_config = dict(run.config)
            
            # Create a copy of the config that we can modify
            sweep_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            
            # Update our config with sweep values
            with open_dict(sweep_cfg):
                for key, value in sweep_config.items():
                    # Handle nested parameters (e.g., model.learning_rate)
                    if "." in key:
                        parts = key.split(".")
                        current = sweep_cfg
                        for part in parts[:-1]:
                            if part.isdigit():  # Handle list indices like hidden_sizes.0
                                part = int(part)
                            current = current[part]
                        current[parts[-1]] = value
                    else:
                        sweep_cfg[key] = value
            
            # Run the training with the updated config
            run_single_training(sweep_cfg, wandb_run=run)
    
    # Start the sweep agent
    wandb.agent(
        cfg.wandb.sweep.id,
        function=sweep_train,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity
    )

def run_single_training(cfg: DictConfig, wandb_run=None):
    """Run a single training with the given config"""
    # Reproducibility
    set_seed(cfg.training.seed)

    # Set float32 matmul precision
    torch.set_float32_matmul_precision(cfg.training.accelerator.float32_matmul_precision)

    # Data setup
    data_dir = Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir
    data_module = CATHeDataModule(
        data_dir=data_dir,
        **{k: v for k, v in cfg.data.items() if k not in ["data_dir", "embedding_dim"]},
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )
    data_module.setup()
    
    # Model setup
    model = CATHeClassifier(
        embedding_dim=cfg.model.embedding_dim,
        hidden_sizes=cfg.model.hidden_sizes,
        num_classes=data_module.num_classes,
        dropout=cfg.model.dropout,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        lr_scheduler=cfg.model.lr_scheduler
    )
    
    # Wandb Logger Setup - use existing run if provided (for sweeps)
    if wandb_run:
        wandb_logger = WandbLogger(experiment=wandb_run)
    else:
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
    log.info("Starting final testing phase...")
    trainer.test(model, data_module, ckpt_path="best")
    log.info("Testing completed - check W&B dashboard for metrics")
    
    # Only finish the W&B run if we created it (not for sweep agents)
    if not wandb_run:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main() 