import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from models.classifier import CATHeClassifier
from utils import get_logger

# Use centralized logger from utils.py
log = get_logger()

def load_embeddings(embeddings_path: Union[str, Path]) -> np.ndarray:
    """Load protein embeddings from an NPZ file.

    Args:
        embeddings_path: Path to the NPZ file containing embeddings.

    Returns:
        Protein embeddings array.

    Raises:
        FileNotFoundError: If the embeddings file doesn't exist.
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
    """Run inference on protein embeddings using the provided model.

    Args:
        model: Trained CATH classifier model.
        embeddings: Input embeddings to predict on.
        batch_size: Batch size for inference.
        device: Device to run inference on ('cuda' or 'cpu').

    Returns:
        Array of predicted CATH superfamily classes.
    """
    model.eval()
    model.to(device)
    all_predictions = []

    with torch.no_grad(), Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        *Progress.get_default_columns(),
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Running inference...", total=len(embeddings))

        for i in range(0, len(embeddings), batch_size):
            batch = torch.FloatTensor(embeddings[i:i + batch_size]).to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())  # Append to list
            progress.update(task, advance=batch.shape[0])

    return torch.cat(all_predictions).numpy() # Efficient concatenation


def save_predictions(
    predictions: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """Save predictions to a CSV file.

    Args:
        predictions: Array of predicted classes.
        output_path: Path to save the predictions CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'predicted_class': predictions})
    df.to_csv(output_path, index=False)
    log.info(f"Predictions saved to {output_path}")

def main(args: argparse.Namespace) -> None:
    """Run the inference pipeline.

    Args:
        args: Command-line arguments.
    """
    try:
        log.info(f"Loading embeddings from {args.embeddings}")
        embeddings = load_embeddings(args.embeddings)

        log.info(f"Loading model from {args.checkpoint}")
        # Load the model, handling device mapping
        model = CATHeClassifier.load_from_checkpoint(
            args.checkpoint,
            map_location=args.device
        )


        # Check if the specified device is available
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            log.warning(f"CUDA device requested ({args.device}), but CUDA is not available. Falling back to CPU.")
            args.device = 'cpu'

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
        log.exception("An error occurred during inference.")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference with a trained CATHe classifier.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint file (.ckpt).'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to the embeddings NPZ file (.npz).'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save the predictions (default: predictions.csv).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32).'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (cuda/cpu, default: cuda if available).'
    )

    args = parser.parse_args()
    main(args) 