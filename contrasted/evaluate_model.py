import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from model import ContrastiveCATHeModel
from data import EmbeddingDataset
from torch.utils.data import DataLoader
import numpy as np
from metrics import calculate_all_metrics_reference
import json
from pathlib import Path
import pickle

# Setup logging
log = logging.getLogger(__name__)

def generate_and_save_embeddings(model, dataloader, device, output_path, labels_path):
    """
    Generates embeddings and saves them and their labels to disk.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    log.info(f"Generating embeddings for {output_path.stem}...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            projected_embeddings = model(embeddings)
            all_embeddings.append(projected_embeddings.cpu().numpy())
            all_labels.extend(labels) # labels is a list of tuples
            if (i+1) % 200 == 0:
                log.info(f"  Processed {i+1} batches...")

    all_embeddings_np = np.concatenate(all_embeddings)
    
    # Save embeddings and labels
    np.save(output_path, all_embeddings_np)
    with open(labels_path, 'wb') as f:
        pickle.dump(all_labels, f)
        
    log.info(f"Saved embeddings ({all_embeddings_np.shape}) to {output_path}")
    log.info(f"Saved labels ({len(all_labels)}) to {labels_path}")

    return all_embeddings_np, all_labels


def load_or_generate_data(model, loader, device, embed_path, labels_path):
    """
    Helper function to load embeddings and labels if they exist, or generate them if they don't.
    """
    if not embed_path.exists() or not labels_path.exists():
        generate_and_save_embeddings(model, loader, device, embed_path, labels_path)
    
    log.info(f"Loading embeddings from {embed_path} and labels from {labels_path}")
    embeddings = np.load(embed_path)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return embeddings, labels


@hydra.main(version_base=None, config_path="../config", config_name="contrastive")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluate a trained model on a large dataset using a reference-based approach.
    
    This script performs evaluation in two stages to handle large datasets:
    1. Generate and save embeddings for train, val, and test sets to disk.
    2. Load embeddings and compute metrics by comparing val/test sets against the train set.
    """
    log.info("Starting evaluation...")
    log.info(f"Using configuration:\n{OmegaConf.to_yaml(cfg)}")

    if 'ckpt_path' not in cfg or cfg.ckpt_path is None:
        raise ValueError("`ckpt_path` must be specified in the config or command line.")

    # --- Setup output directories ---
    ckpt_name = Path(cfg.ckpt_path).stem
    eval_output_dir = Path(cfg.training.output_dir) / "evaluation_results" / ckpt_name
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Evaluation results will be saved in: {eval_output_dir}")
    
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load Model ---
    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = ContrastiveCATHeModel.load_from_checkpoint(cfg.ckpt_path, map_location=device, cfg=cfg.model)
    model.to(device)

    # --- Load Datasets ---
    log.info("Initializing datasets...")
    data_cfg = cfg.data
    train_dataset = EmbeddingDataset(data_cfg.train_embeddings, data_cfg.train_labels)
    val_dataset = EmbeddingDataset(data_cfg.val_embeddings, data_cfg.val_labels)
    test_dataset = EmbeddingDataset(data_cfg.test_embeddings, data_cfg.test_labels)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    
    # --- STAGE 1: Generate and Save Embeddings ---
    log.info("--- Stage 1: Loading or generating embeddings ---")
    
    # Define paths for embeddings and labels
    train_embed_path = eval_output_dir / "train_embeddings.npy"
    train_labels_path = eval_output_dir / "train_labels.pkl"
    val_embed_path = eval_output_dir / "val_embeddings.npy"
    val_labels_path = eval_output_dir / "val_labels.pkl"
    test_embed_path = eval_output_dir / "test_embeddings.npy"
    test_labels_path = eval_output_dir / "test_labels.pkl"

    # Load or generate all data splits
    train_embeddings, train_labels = load_or_generate_data(model, train_loader, device, train_embed_path, train_labels_path)
    val_embeddings, val_labels = load_or_generate_data(model, val_loader, device, val_embed_path, val_labels_path)
    test_embeddings, test_labels = load_or_generate_data(model, test_loader, device, test_embed_path, test_labels_path)

    # --- STAGE 2: Metric Calculation ---
    log.info("--- Stage 2: Calculating metrics ---")
    results = {}
    
    metric_batch_size = cfg.training.get('metric_batch_size', 1024)

    for level_idx, level_name in enumerate(['C', 'A', 'T', 'H']):
        log.info(f"--- Evaluating for CATH level: {level_name} (level {level_idx}) ---")
        
        # Validation metrics
        log.info("Calculating validation metrics...")
        val_metrics = calculate_all_metrics_reference(
            query_embeddings=val_embeddings,
            query_labels=val_labels,
            lookup_embeddings=train_embeddings,
            lookup_labels=train_labels,
            level=level_idx,
            k=1,
            stage='val',
            batch_size=metric_batch_size
        )
        results[f'val_{level_name}'] = val_metrics
        log.info(f"Validation Metrics ({level_name}): {val_metrics}")

        # Test metrics
        log.info("Calculating test metrics...")
        test_metrics = calculate_all_metrics_reference(
            query_embeddings=test_embeddings,
            query_labels=test_labels,
            lookup_embeddings=train_embeddings,
            lookup_labels=train_labels,
            level=level_idx,
            k=1,
            stage='test',
            batch_size=metric_batch_size
        )
        results[f'test_{level_name}'] = test_metrics
        log.info(f"Test Metrics ({level_name}): {test_metrics}")

    # --- Save Final Results ---
    # Ensure all metric values are JSON-serializable
    serializable_results = {
        key: {m: float(v) for m, v in val.items()}
        for key, val in results.items()
    }

    results_file = eval_output_dir / "final_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
        
    log.info(f"Final evaluation metrics saved to {results_file}")
    log.info("Evaluation finished successfully.")

if __name__ == "__main__":
    evaluate() 