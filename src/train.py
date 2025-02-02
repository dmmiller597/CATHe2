import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
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
            filename='cathe-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=cfg.training.save_top_k,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.early_stopping_patience,
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar()
    ]

def setup_trainer(cfg: DictConfig, callbacks: list) -> pl.Trainer:
    logger = TensorBoardLogger('logs', name='cathe', default_hp_metric=False)
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
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
        precision=cfg.training.get('precision', 32)
    )

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
        
        # Set random seed
        set_seed(cfg.training.get('seed', 42), workers=True)
        
        # Initialize data module
        log.info("Initializing data module...")
        data_module = CATHeDataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            train_embeddings=cfg.data.train_embeddings,
            val_embeddings=cfg.data.val_embeddings,
            test_embeddings=cfg.data.test_embeddings,
            train_labels=cfg.data.train_labels,
            val_labels=cfg.data.val_labels,
            test_labels=cfg.data.test_labels,
            num_workers=cfg.training.get('num_workers', 4)
        )
        
        # Initialize model
        log.info("Initializing model...")
        model = CATHeClassifier(
            embedding_dim=cfg.model.embedding_dim,
            hidden_sizes=cfg.model.hidden_sizes,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.get('dropout', 0.2),
            learning_rate=cfg.training.learning_rate,
            use_batch_norm=cfg.model.get('use_batch_norm', True)
        )
        
        # Setup trainer
        trainer = setup_trainer(cfg, setup_callbacks(cfg))
        
        # Train the model
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
    
    except Exception as e:
        log.exception("An error occurred during training")
        raise

if __name__ == '__main__':
    main() 