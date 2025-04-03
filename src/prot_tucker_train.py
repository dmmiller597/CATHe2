import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch
from pathlib import Path
import logging # Use standard logging

# Import the new model
from utils import get_logger, set_seed
from data.contrastive_data_module import ContrastiveDataModule
# from models.contrasted import ContrastiveCATHeModel # Remove old model import
from models.prot_tucker_lightning import ProtTuckerLightning # Add new model import

# Use standard logging setup if not already done
log = get_logger() # Assumes get_logger is defined in utils

# Change config_name here
@hydra.main(config_path="../config", config_name="prot_tucker", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main training function for ProtTuckerLightning model."""
    # Reproducibility
    if cfg.training.seed is not None:
        log.info(f"Setting seed: {cfg.training.seed}")
        set_seed(cfg.training.seed)
    else:
        log.info("No seed provided, training will not be deterministic.")

    # Set float32 matmul precision if configured and using torch >= 1.12
    if hasattr(cfg.training.accelerator, "float32_matmul_precision") and hasattr(torch, 'set_float32_matmul_precision'):
        log.info(f"Setting float32 matmul precision: {cfg.training.accelerator.float32_matmul_precision}")
        torch.set_float32_matmul_precision(cfg.training.accelerator.float32_matmul_precision)

    # Data setup - Instantiate ContrastiveDataModule using Hydra config
    log.info(f"Instantiating DataModule: {cfg.data._target_}")
    # Pass relevant parts of config, Hydra handles _target_
    data_module = hydra.utils.instantiate(cfg.data)
    # Manually call setup to ensure dimensions are available IF needed before model init
    # data_module.setup('fit') # Usually Trainer calls setup, but needed if model needs data dims

    # Model setup - Instantiate ProtTuckerLightning using Hydra config
    log.info(f"Instantiating Model: {cfg.model._target_}")
    # Ensure input_embedding_dim matches data if not explicitly set/validated
    if 'input_embedding_dim' not in cfg.model:
         data_module.setup('fit') # Ensure train dataset is loaded
         if data_module.embedding_dim:
             log.info(f"Inferring input_embedding_dim from data: {data_module.embedding_dim}")
             cfg.model.input_embedding_dim = data_module.embedding_dim
         else:
             raise ValueError("Could not infer input_embedding_dim from data, please set it in the config.")

    model = hydra.utils.instantiate(cfg.model) # Pass model config section

    # Callbacks - Instantiate using Hydra config
    callbacks = []
    if cfg.callbacks:
        for _, cb_conf in cfg.callbacks.items():
            log.info(f"Instantiating Callback: {cb_conf._target_}")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    else:
        log.warning("No callbacks configured.")


    # Wandb Logger - Instantiate using Hydra config
    wandb_logger = None
    if cfg.logger:
         log.info(f"Instantiating Logger: {cfg.logger._target_}")
         # Pass the whole config for logging hyperparams
         wandb_logger = hydra.utils.instantiate(cfg.logger, config=OmegaConf.to_container(cfg, resolve=True))
         # Watch model gradients/parameters (optional)
         # wandb_logger.watch(model, log="all", log_freq=cfg.training.log_every_n_steps)
    else:
        log.warning("No logger configured.")

    # Trainer - Instantiate using Hydra config
    log.info("Instantiating Trainer...")
    trainer_params = { # Extract relevant trainer args from config
        'accelerator': cfg.training.accelerator.devices if hasattr(cfg.training.accelerator, 'devices') else 'auto', # Handle simpler config
        'devices': cfg.training.accelerator.devices if hasattr(cfg.training.accelerator, 'devices') else 'auto',
        'max_epochs': cfg.training.max_epochs,
        'callbacks': callbacks,
        'logger': wandb_logger,
        'deterministic': cfg.training.seed is not None,
        'gradient_clip_val': cfg.training.gradient_clip_val if hasattr(cfg.training, 'gradient_clip_val') else None,
        'accumulate_grad_batches': cfg.training.accumulate_grad_batches,
        'precision': cfg.training.precision,
        'log_every_n_steps': cfg.training.log_every_n_steps,
        'enable_progress_bar': True, # Assuming you want progress bar
        'enable_model_summary': True, # Assuming you want summary
        'default_root_dir': hydra.utils.get_original_cwd() / cfg.training.output_dir if hasattr(cfg.training, 'output_dir') else None,
    }
    # Remove None values before passing to Trainer
    trainer_params = {k: v for k, v in trainer_params.items() if v is not None}
    trainer = pl.Trainer(**trainer_params)

    # Training
    log.info("Starting training...")
    trainer.fit(model, data_module)
    log.info("Training finished.")

    # Final testing (optional, requires test data)
    if cfg.data.test_embeddings and cfg.data.test_labels:
        log.info("Starting final testing phase...")
        # Load best checkpoint based on monitored metric
        best_ckpt_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "best"
        if best_ckpt_path:
             log.info(f"Loading best checkpoint for testing: {best_ckpt_path}")
             trainer.test(model, data_module, ckpt_path=best_ckpt_path)
             log.info("Testing completed.")
        else:
             log.warning("Could not find best checkpoint path for testing.")
    else:
        log.info("Test data not provided, skipping final testing.")

    # Finish Wandb run
    if wandb_logger:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()