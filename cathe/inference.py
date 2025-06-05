import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel
import re
from tqdm import tqdm
import logging
from rich.logging import RichHandler
import warnings
warnings.filterwarnings('ignore')

from model import CATHeClassifier
from utils import get_logger

log = get_logger()

class ProtT5Embedder:
    """Generate ProtT5 embeddings for protein sequences."""
    
    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc", device: str = "auto") -> None:
        """Initialize ProtT5 model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier for ProtT5
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        log.info(f"Loading ProtT5 model on {self.device}")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            log.info("ProtT5 model loaded successfully")
        except Exception as e:
            log.error(f"Failed to load ProtT5 model: {e}")
            raise
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for ProtT5."""
        # Remove invalid characters and add spaces between amino acids
        sequence = re.sub(r"[UZOB]", "X", sequence.upper())
        return " ".join(list(sequence))
    
    def embed_sequence(self, sequence: str, max_length: int = 1024) -> np.ndarray:
        """Generate embedding for a single protein sequence.
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Per-residue embeddings averaged to sequence-level embedding
        """
        processed_seq = self._preprocess_sequence(sequence)
        
        inputs = self.tokenizer(
            processed_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Average embeddings over sequence length (excluding padding)
            sequence_embedding = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
        return sequence_embedding.cpu().numpy().squeeze()
    
    def embed_batch(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        """Generate embeddings for a batch of sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Number of sequences to process at once
            
        Returns:
            Array of sequence embeddings
        """
        embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
            batch = sequences[i:i + batch_size]
            batch_embeddings = []
            
            for seq in batch:
                try:
                    emb = self.embed_sequence(seq)
                    batch_embeddings.append(emb)
                except Exception as e:
                    log.warning(f"Failed to embed sequence (length {len(seq)}): {e}")
                    # Use zero embedding as fallback
                    batch_embeddings.append(np.zeros(1024))  # ProtT5-XL has 1024 dimensions
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

class StructureClassifier:
    """CATH superfamily classifier using trained CATHe model."""
    
    def __init__(self, checkpoint_path: str, label_encoder_path: Optional[str] = None) -> None:
        """Initialize classifier with trained model.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            label_encoder_path: Path to saved label encoder (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            log.info(f"Loading model from {checkpoint_path}")
            self.model = CATHeClassifier.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            self.model.to(self.device)
        except Exception as e:
            log.error(f"Failed to load model checkpoint: {e}")
            raise
        
        # Load label encoder if provided
        self.label_encoder = None
        if label_encoder_path and Path(label_encoder_path).exists():
            try:
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                    log.info(f"Loaded label encoder with {len(self.label_encoder.classes_)} classes")
            except Exception as e:
                log.error(f"Failed to load label encoder: {e}")
                raise
        else:
            log.warning("No label encoder provided - will return numeric predictions")
    
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on protein embeddings.
        
        Args:
            embeddings: Array of protein embeddings
            
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(embeddings_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
        
        return predicted_classes.cpu().numpy(), confidence_scores.cpu().numpy()
    
    def predict_with_labels(self, embeddings: np.ndarray) -> List[Dict[str, any]]:
        """Make predictions and return with superfamily labels if available.
        
        Args:
            embeddings: Array of protein embeddings
            
        Returns:
            List of prediction dictionaries
        """
        predicted_classes, confidence_scores = self.predict(embeddings)
        
        results = []
        for pred_class, confidence in zip(predicted_classes, confidence_scores):
            result = {
                "predicted_class_id": int(pred_class),
                "confidence": float(confidence),
                "predicted_superfamily": None
            }
            
            if self.label_encoder:
                try:
                    result["predicted_superfamily"] = self.label_encoder.inverse_transform([pred_class])[0]
                except Exception as e:
                    log.warning(f"Failed to decode class {pred_class}: {e}")
                    result["predicted_superfamily"] = f"Unknown_Class_{pred_class}"
            
            results.append(result)
        
        return results

def load_fasta_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    """Load protein sequences from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        List of (sequence_id, sequence) tuples
    """
    sequences = []
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append((record.id, str(record.seq)))
        log.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    except Exception as e:
        log.error(f"Error loading FASTA file: {e}")
        raise ValueError(f"Failed to parse FASTA file {fasta_path}: {str(e)}")
    
    return sequences

def save_predictions(
    predictions: List[Dict[str, any]], 
    sequence_ids: List[str], 
    output_path: str
) -> None:
    """Save predictions to CSV file.
    
    Args:
        predictions: List of prediction dictionaries
        sequence_ids: List of sequence identifiers
        output_path: Path to save results
    """
    try:
        results_df = pd.DataFrame(predictions)
        results_df.insert(0, "sequence_id", sequence_ids)
        results_df.to_csv(output_path, index=False)
        log.info(f"Predictions saved to {output_path}")
    except Exception as e:
        log.error(f"Failed to save predictions: {e}")
        raise

def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Classify protein structures using trained CATHe model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--fasta", "-f",
        required=True,
        help="Path to input FASTA file"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to trained model checkpoint (.ckpt file)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="predictions.csv",
        help="Output CSV file for predictions"
    )
    
    parser.add_argument(
        "--label_encoder",
        help="Path to saved label encoder pickle file"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for ProtT5"
    )
    
    return parser

def main() -> None:
    """Main inference pipeline."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.fasta).exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    log.info("Starting protein structure classification pipeline")
    
    try:
        # Step 1: Load sequences
        log.info("Loading protein sequences...")
        sequences_data = load_fasta_sequences(args.fasta)
        sequence_ids, sequences = zip(*sequences_data)
        
        # Step 2: Generate embeddings
        log.info("Generating ProtT5 embeddings...")
        embedder = ProtT5Embedder(device=args.device)
        embeddings = embedder.embed_batch(sequences, batch_size=args.batch_size)
        
        # Step 3: Load classifier and make predictions
        log.info("Loading classifier and making predictions...")
        classifier = StructureClassifier(args.checkpoint, args.label_encoder)
        predictions = classifier.predict_with_labels(embeddings)
        
        # Step 4: Save results
        log.info("Saving predictions...")
        save_predictions(predictions, sequence_ids, args.output)
        
        # Print summary
        log.info(f"Processed {len(sequences)} sequences")
        if classifier.label_encoder:
            unique_predictions = set(p["predicted_superfamily"] for p in predictions)
            log.info(f"Predicted {len(unique_predictions)} unique superfamilies")
        
        log.info("Pipeline completed successfully!")
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
