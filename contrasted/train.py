import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch
from pathlib import Path
import os # Import os

from utils import get_logger, set_seed
from contrastive_data_module import CATHeDataModule, CATH_LEVEL_NAMES
from contrasted import ContrastiveCATHeModel

log = get_logger()

@hydra.main(config_path="../config", config_name="contrastive", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main training function for hierarchical contrastive learning model."""
    base_output_dir = Path(cfg.training.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Base output directory: {base_output_dir}")

    # Global settings
    set_seed(cfg.training.seed)
    torch.set_float32_matmul_precision(cfg.training.accelerator.float32_matmul_precision)

    # --- Hierarchical Training Loop ---
    cath_levels_to_train = cfg.training.cath_levels
    log.info(f"Starting hierarchical training for CATH levels: {cath_levels_to_train}")

    last_best_checkpoint_path = None # Track the best checkpoint from the previous level

    for level_idx, cath_level in enumerate(cath_levels_to_train):
        level_name = CATH_LEVEL_NAMES.get(cath_level, f"Level_{cath_level}")
        log.info(f"\n{'='*20} Starting Training for CATH Level {cath_level} ({level_name}) {'='*20}")

        # --- Level-Specific Configuration ---
        level_output_dir = base_output_dir / f"level_{cath_level}_{level_name}"
        level_ckpt_dir = level_output_dir / "checkpoints"
        level_log_dir = level_output_dir / "logs"
        level_ckpt_dir.mkdir(parents=True, exist_ok=True)
        level_log_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Output directory for level {level_name}: {level_output_dir}")

        # --- Data Setup for Current Level ---
        data_module = CATHeDataModule(
            data_dir=Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir,
            train_embeddings_file=cfg.data.train_embeddings,
            train_labels_file=cfg.data.train_labels,
            val_embeddings_file=cfg.data.val_embeddings,
            val_labels_file=cfg.data.val_labels,
            test_embeddings_file=getattr(cfg.data, "test_embeddings", None), # Use getattr for safety
            test_labels_file=getattr(cfg.data, "test_labels", None),
            cath_level=cath_level, # Pass the current level
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )
        # It's crucial to call setup to load data and determine num_classes, etc.
        data_module.setup('fit') # Setup for fit stage initially

        # --- Model Setup for Current Level ---
        if last_best_checkpoint_path is None:
            log.info("Initializing model from scratch for the first level.")
            model = ContrastiveCATHeModel(
                input_embedding_dim=cfg.data.embedding_dim, # Should match data_module.embedding_dim
                projection_hidden_dims=cfg.model.projection_dims,
                output_embedding_dim=cfg.model.output_dim,
                dropout=cfg.model.dropout,
                learning_rate=cfg.model.learning_rate,
                weight_decay=cfg.model.weight_decay,
                knn_val_neighbors=cfg.model.n_neighbors,
                val_max_samples=cfg.model.val_max_samples,
                warmup_epochs=cfg.model.warmup_epochs,
                warmup_start_factor=cfg.model.warmup_start_factor,
                visualization_method=cfg.model.visualization_method,
                # Pass level-specific vis dirs
                tsne_viz_dir=str(level_output_dir / "tsne_plots"),
                umap_viz_dir=str(level_output_dir / "umap_plots"),
                lr_scheduler_config={ # Pass the scheduler config dict
                    "monitor": cfg.model.lr_scheduler.monitor,
                    "mode": cfg.model.lr_scheduler.mode,
                    "factor": cfg.model.lr_scheduler.factor,
                    "patience": cfg.model.lr_scheduler.patience,
                    "min_lr": cfg.model.lr_scheduler.min_lr
                },
                 # Pass num_classes and decoder for potential internal use/logging
                 # num_classes=data_module.get_num_classes(), # Optional: if model uses it
                 # label_decoder=data_module.get_label_decoder() # Optional: if model uses it
            )
        else:
            log.info(f"Loading model from previous level's best checkpoint: {last_best_checkpoint_path}")
            # When loading, PL tries to restore hparams. Pass necessary ones that might
            # have changed or are needed by the __init__ logic again.
            # Especially paths for visualization need updating.
            model = ContrastiveCATHeModel.load_from_checkpoint(
                last_best_checkpoint_path,
                # Provide hparams that might need overriding or are essential for re-init logic
                strict=False, # Be less strict if some hparams mismatch (e.g., num_classes)
                # --- Update paths and potentially LR ---
                learning_rate=cfg.model.learning_rate, # Reset LR for the new phase? Or let scheduler continue? Resetting is safer.
                tsne_viz_dir=str(level_output_dir / "tsne_plots"),
                umap_viz_dir=str(level_output_dir / "umap_plots"),
                lr_scheduler_config={ # Ensure scheduler config is passed again
                     "monitor": cfg.model.lr_scheduler.monitor,
                     "mode": cfg.model.lr_scheduler.mode,
                     "factor": cfg.model.lr_scheduler.factor,
                     "patience": cfg.model.lr_scheduler.patience,
                     "min_lr": cfg.model.lr_scheduler.min_lr
                 },
                 # Optional: Update other potentially changed hparams if needed
                 # num_classes=data_module.get_num_classes(),
                 # label_decoder=data_module.get_label_decoder()
            )
            log.info("Model loaded successfully.")


        # --- Callbacks for Current Level ---
        # Checkpoint callback: Monitors the metric for *this* level
        checkpoint_callback = ModelCheckpoint(
            dirpath=level_ckpt_dir,
            filename=f'{level_name}-{{epoch:02d}}-{{val/knn_balanced_acc:.4f}}', # Include level name
            monitor=cfg.training.monitor_metric, # Still 'val/knn_balanced_acc'
            mode=cfg.training.monitor_mode,
            save_last=True,
            save_top_k=1, # Save only the best one
            auto_insert_metric_name=False
        )
        # Early stopping callback: Monitors the metric for *this* level
        early_stopping_callback = EarlyStopping(
            monitor=cfg.training.monitor_metric,
            patience=cfg.training.early_stopping_patience,
            mode=cfg.training.monitor_mode,
            verbose=True, # Log when stopping
            check_on_train_epoch_end=False # Explicitly check after validation epoch
        )
        # Standard callbacks
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        progress_bar = RichProgressBar()

        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
            progress_bar
        ]

        # --- Logger for Current Level ---
        # Use groups in Wandb to keep levels organized under one experiment if desired
        # Or create separate runs per level
        wandb_logger = WandbLogger(
            project="CATHe-Contrastive-Hierarchical", # Or use existing project
            name=f"Level_{cath_level}_{level_name}", # Run name for this level
            save_dir=str(level_log_dir), # Log files saved here
            # group=f"HierarchicalRun_{cfg.training.seed}", # Optional: Group runs
            log_model=True, # Log checkpoints to Wandb
            config=OmegaConf.to_container(cfg, resolve=True) # Log full config
        )
        # Log the current level being trained
        wandb_logger.experiment.config.update({"current_cath_level": cath_level, "current_cath_level_name": level_name})


        # --- Trainer for Current Level ---
        # Create a new trainer for each level to reset optimizer state, epoch count, etc.
        trainer = pl.Trainer(
            accelerator='auto',
            devices=cfg.training.accelerator.devices,
            max_epochs=cfg.training.max_epochs,
            callbacks=callbacks,
            logger=wandb_logger,
            # deterministic=cfg.training.seed is not None, # Use pl.seed_everything instead generally
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            precision=cfg.training.precision,
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_progress_bar=True, # Already handled by RichProgressBar callback
            enable_model_summary=True,
            default_root_dir=str(level_output_dir), # Sets base dir for logs/checkpoints if not specified elsewhere
            num_sanity_val_steps=2 # Run a couple of sanity checks
        )

        # --- Training for Current Level ---
        log.info(f"Starting trainer.fit for CATH Level {cath_level} ({level_name})...")
        trainer.fit(model, datamodule=data_module) # Pass datamodule=data_module

        # Store the best checkpoint path for the next level
        # Ensure the callback actually saved something (might not if training is very short/fails)
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
             last_best_checkpoint_path = checkpoint_callback.best_model_path
             log.info(f"Best checkpoint for level {level_name} saved at: {last_best_checkpoint_path}")
        else:
             log.warning(f"No best checkpoint path found for level {level_name}. Using last known checkpoint if available.")
             # Fallback to last checkpoint if 'best' isn't available (e.g., KeyboardInterrupt)
             if checkpoint_callback.last_model_path and os.path.exists(checkpoint_callback.last_model_path):
                 last_best_checkpoint_path = checkpoint_callback.last_model_path
                 log.info(f"Using last checkpoint for level {level_name}: {last_best_checkpoint_path}")
             else:
                 log.error(f"Cannot proceed to next level: No checkpoint saved for level {level_name}.")
                 break # Stop the hierarchical process if a level fails to produce a checkpoint


        # --- Optional Testing for Current Level ---
        if data_module.test_dataloader() is not None: # Check if test data exists for this level
            log.info(f"Starting final testing for level {level_name} using checkpoint: {last_best_checkpoint_path}")
            # Reload the best model for testing explicitly if needed, though Trainer might handle it with ckpt_path='best' logic internally
            # It's safer to pass the explicit path from the just-finished training run.
            test_results = trainer.test(model=model, datamodule=data_module, ckpt_path=last_best_checkpoint_path)
            log.info(f"Testing results for level {level_name}: {test_results}")
            # Log test metrics to wandb if needed
            # wandb_logger.log_metrics({f"test/{k}_level{cath_level}": v for k, v in test_results[0].items()})
        else:
            log.info(f"No test data found or configured for level {level_name}. Skipping testing.")

        # --- Finish Wandb Run for the Level ---
        # Finish the run here if you want separate runs per level
        # Or keep it running if using groups for a single logical experiment
        wandb_logger.experiment.finish()
        log.info(f"Finished training and testing for CATH Level {cath_level} ({level_name}).")

    log.info("Hierarchical training finished for all specified CATH levels.")


if __name__ == "__main__":
    main() 