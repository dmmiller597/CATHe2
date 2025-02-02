import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from models.classifier import CATHeClassifier
from utils import get_logger

# Use centralized logger from utils.py
log = get_logger()

def load_embeddings(embeddings_path: Union[str, Path]) -> np.ndarray:
    """Load protein embeddings from NPZ file.
    
    Args:
        embeddings_path: Path to NPZ file containing embeddings
        
    Returns:
        Protein embeddings array
        
    Raises:
        FileNotFoundError: If embeddings file doesn't exist
    """
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
    with np.load(embeddings_path) as data:
        return data['arr_0']

def predict(
    model: CATHeClassifier,
    embeddings: np.ndarray,
    batch_size: int = 32,
    device: str = 'cuda'
) -> np.ndarray:
    """Run inference on protein embeddings.
    
    Args:
        model: Trained CATH classifier model
        embeddings: Input embeddings to predict on
        batch_size: Batch size for inference (default: 32)
        device: Device to run inference on (default: 'cuda')
        
    Returns:
        Array of predicted CATH superfamily classes
    """
    model.eval()
    model = model.to(device)
    predictions = []
    
    with torch.no_grad(), Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Running inference...", total=len(embeddings))
        
        for i in range(0, len(embeddings), batch_size):
            batch = torch.FloatTensor(embeddings[i:i + batch_size]).to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            progress.update(task, advance=len(batch))
    
    return np.array(predictions)

def save_predictions(
    predictions: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """Save predictions to CSV file.
    
    Args:
        predictions: Array of predicted classes
        output_path: Path to save predictions CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'predicted_class': predictions}).to_csv(output_path, index=False)
    log.info(f"Predictions saved to {output_path}")

def main(args: argparse.Namespace) -> None:
    """Run inference pipeline.
    
    Args:
        args: Command line arguments
    """
    try:
        log.info(f"Loading embeddings from {args.embeddings}")
        embeddings = load_embeddings(args.embeddings)
        
        log.info(f"Loading model from {args.checkpoint}")
        model = CATHeClassifier.load_from_checkpoint(
            args.checkpoint,
            map_location=args.device
        )
        
        log.info("Running inference...")
        predictions = predict(
            model,
            embeddings,
            batch_size=args.batch_size,
            device=args.device
        )
        
        save_predictions(predictions, args.output)
        log.info("Inference completed successfully!")
        
    except Exception as e:
        log.exception("An error occurred during inference")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference with trained CATHe classifier'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to embeddings NPZ file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (cuda/cpu)'
    )
    
    args = parser.parse_args()
    main(args) 