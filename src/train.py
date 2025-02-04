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
    return [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='cathe-{epoch:02d}-{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=cfg.training.save_top_k,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_acc',
            patience=cfg.training.early_stopping_patience,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar()
    ]

def setup_trainer(cfg: DictConfig, callbacks: list) -> pl.Trainer:
    logger = WandbLogger(
        project="CATHe",
        log_model=True
    )
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    accelerator = 'gpu' if torch.cuda.is_available() and cfg.training.gpus > 0 else 'cpu'
    devices = cfg.training.gpus if accelerator == 'gpu' else None
    log.info(f"Training on {devices if devices else 1} {accelerator}(s)")
    
    return pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        enable_progress_bar=True,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_checkpointing=True,
        enable_model_summary=True
    )

@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Get original working directory
    original_cwd = hydra.utils.get_original_cwd()
    
    # Use Path to handle paths relative to original working directory
    project_root = Path(original_cwd)
    
    set_seed(cfg.training.seed)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    
    # Initialize data module with absolute path
    data_module = CATHeDataModule(
        data_dir=str(project_root / cfg.data.data_dir),
        train_embeddings=cfg.data.train_embeddings,
        train_labels=cfg.data.train_labels,
        val_embeddings=cfg.data.val_embeddings,
        val_labels=cfg.data.val_labels,
        test_embeddings=cfg.data.test_embeddings,
        test_labels=cfg.data.test_labels,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )
    
    # Setup the data module and determine num_classes
    data_module.setup()
    
    # Disable structured config so we can add new keys
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.num_classes = data_module.num_classes
    
    # Initialize model with updated config parameters
    model = CATHeClassifier(**cfg.model)
    
    # Setup trainer and start training
    trainer = setup_trainer(cfg, setup_callbacks(cfg))
    
    log.info("Starting training...")
    trainer.fit(model, data_module)
    
    if trainer.checkpoint_callback.best_model_path:
        log.info(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
        log.info("Testing best model...")
        trainer.test(model, data_module)
        
        # Save test metrics
        test_results = Path('logs') / 'test_results.yaml'
        with open(test_results, 'w') as f:
            import yaml
            yaml.dump(trainer.callback_metrics, f)
        log.info(f"Test results saved to {test_results}")
    
    log.info("Training completed successfully!")

if __name__ == "__main__":
    main() 