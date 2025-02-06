import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from rich.progress import Progress
from models.classifier import CATHeClassifier
from utils import get_logger
import torchmetrics

# Use centralized logger from utils.py
log = get_logger()

def load_embeddings(embeddings_path: Union[str, Path]) -> torch.Tensor:
    """Load protein embeddings from an NPZ file.

    Args:
        embeddings_path: Path to the NPZ file containing embeddings.

    Returns:
        Protein embeddings tensor.

    Raises:
        FileNotFoundError: If the embeddings file doesn't exist.
    """
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    with np.load(embeddings_path) as data:
        embeddings = torch.from_numpy(data['arr_0']).float() # Load directly as float32 tensor
        return embeddings

def predict(
    model: CATHeClassifier,
    embeddings: torch.Tensor,
    batch_size: int = 32,
    device: str = 'cuda'
) -> torch.Tensor:
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
    num_embeddings = embeddings.shape[0]
    all_predictions = torch.empty(num_embeddings, dtype=torch.long) # Pre-allocate

    with torch.no_grad(), Progress(transient=True) as progress: # Simplified progress
        task = progress.add_task("[cyan]Running inference...", total=num_embeddings)

        for i in range(0, num_embeddings, batch_size):
            batch = embeddings[i:i + batch_size].to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            all_predictions[i:i+batch_size] = preds.cpu() # Fill pre-allocated tensor
            progress.update(task, advance=batch.shape[0])

    return all_predictions

def save_predictions(
    predictions: torch.Tensor,
    output_path: Union[str, Path]
) -> None:
    """Save predictions to a CSV file.

    Args:
        predictions: Array of predicted classes.
        output_path: Path to save the predictions CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'predicted_class': predictions.numpy()})
    df.to_csv(output_path, index=False)
    log.info(f"Predictions saved to {output_path}")

def load_annotations(annotations_path: Union[str, Path]) -> torch.Tensor:
    """Load ground truth annotations from a CSV file.

    Args:
        annotations_path: Path to the CSV file containing annotations.

    Returns:
        Tensor of ground truth classes.

    Raises:
        FileNotFoundError: If the annotations file doesn't exist.
    """
    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    df = pd.read_csv(annotations_path)
    # Assuming the ground truth SF is in a column named 'SF'
    codes = pd.Categorical(df['SF']).codes
    return torch.tensor(codes, dtype=torch.long)

def evaluate_predictions(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    num_classes: int,
) -> None:
    """Evaluate predictions against ground truth and log metrics.

    Args:
        predictions: Array of predicted classes.
        ground_truth: Array of ground truth classes.
        num_classes: The total number of classes.
    """

    # Ensure predictions and ground truth are on the CPU
    predictions = predictions.to('cpu')
    ground_truth = ground_truth.to('cpu')

    # Initialize metrics ONCE
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
    mcc = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=num_classes)

    # Calculate metrics
    acc = accuracy(predictions, ground_truth)
    f1 = f1_score(predictions, ground_truth)
    mcc_val = mcc(predictions, ground_truth)

    # Log metrics
    log.info(f"Accuracy: {acc:.4f}")
    log.info(f"F1 Score (Macro): {f1:.4f}")
    log.info(f"MCC: {mcc_val:.4f}")

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
        num_classes = model.hparams.num_classes

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

        if args.annotations:
            log.info(f"Loading annotations from {args.annotations}")
            ground_truth = load_annotations(args.annotations)
            if len(predictions) != len(ground_truth):
                raise ValueError(
                    "Number of predictions and ground truth annotations do not match."
                )
            evaluate_predictions(predictions, ground_truth, num_classes)

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
    parser.add_argument(
        '--annotations',
        type=str,
        default=None,
        help='Path to the CSV file containing ground truth annotations (optional).'
    )

    args = parser.parse_args()
    if args.annotations and not Path(args.annotations).exists():
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")
    main(args) 