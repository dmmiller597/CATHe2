#!/usr/bin/env python3
"""
CATHe Protein Structure Classification - Inference Pipeline

Predicts CATH superfamilies for protein sequences using ProtT5 + trained CATHe model.

USAGE:
    python cathe/inference.py -f proteins.fasta -c model.ckpt -t train_labels.csv

REQUIRED FILES:
    proteins.fasta     - Input protein sequences (FASTA format)
    model.ckpt         - Trained CATHe PyTorch Lightning checkpoint  
    train_labels.csv   - Training labels with 'SF' column (CATH IDs)

OUTPUT:
    predictions.csv    - Results: sequence_id, cath_id, predicted_class_id, confidence

REQUIREMENTS:
    - 8GB+ GPU memory, source venv/bin/activate
    
EXAMPLE:
    python cathe/inference.py \
        --fasta /SAN/orengolab/cath_alphafold/t_level_clustering/3.30.450_cluster_0.8tmscore_0.62_cov_0.73_rep_seq.fasta \
        --checkpoint outputs_cathe/checkpoints/last.ckpt \
        --cache_dir /SAN/orengolab/functional-families/CATHe2/model_cache \
        --training_labels data/TED/s30/protT5/protT5_labels_train.csv
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
import os
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel
import re
from tqdm import tqdm
import logging
from rich.logging import RichHandler
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

from model import CATHeClassifier
from utils import get_logger

log = get_logger()

class ProtT5Embedder:
    """Generate ProtT5 embeddings for protein sequences."""
    
    def __init__(
        self, 
        model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc", 
        device: str = "auto"
    ) -> None:
        """Initialize ProtT5 model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
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
    
    def __init__(self, checkpoint_path: str, training_labels_path: str) -> None:
        """Initialize classifier with trained model and create label encoder.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            training_labels_path: Path to training labels CSV file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        try:
            log.info(f"Loading model from: {checkpoint_path}")
            self.model = CATHeClassifier.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            self.model.to(self.device)
            self.num_classes = self.model.num_classes
            log.info(f"Model expects {self.num_classes} classes")
            
        except Exception as e:
            log.error(f"Failed to load model checkpoint: {e}")
            raise
        
        # Create label encoder from training data
        self.label_encoder = self._create_label_encoder(training_labels_path)
        
        log.info(f"✓ Classifier ready with {self.num_classes} CATH superfamilies")
    
    def _create_label_encoder(self, training_labels_path: str) -> LabelEncoder:
        """Create label encoder from training data."""
        log.info(f"Creating label encoder from: {training_labels_path}")
        
        if not Path(training_labels_path).exists():
            raise FileNotFoundError(f"Training labels file not found: {training_labels_path}")
        
        # Load training labels and create encoder
        df = pd.read_csv(training_labels_path)
        if 'SF' not in df.columns:
            raise ValueError(f"Training labels CSV must contain 'SF' column. Found: {df.columns.tolist()}")
        
        label_encoder = LabelEncoder().fit(df['SF'])
        
        # Validate against model
        if len(label_encoder.classes_) != self.num_classes:
            raise ValueError(
                f"Label encoder mismatch! Model expects {self.num_classes} classes, "
                f"but training data has {len(label_encoder.classes_)} classes"
            )
        
        log.info(f"Label encoder created with {len(label_encoder.classes_)} classes")
        return label_encoder
    
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on protein embeddings.
        
        Args:
            embeddings: Array of protein embeddings
            
        Returns:
            Tuple of (predicted_class_ids, confidence_scores)
        """
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(embeddings_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
        
        return predicted_classes.cpu().numpy(), confidence_scores.cpu().numpy()
    
    def predict_with_cath_labels(self, embeddings: np.ndarray) -> List[Dict[str, any]]:
        """Make predictions and return with CATH superfamily labels.
        
        Args:
            embeddings: Array of protein embeddings
            
        Returns:
            List of prediction dictionaries with CATH superfamily IDs
        """
        predicted_classes, confidence_scores = self.predict(embeddings)
        
        # Decode to CATH superfamily IDs
        cath_ids = self.label_encoder.inverse_transform(predicted_classes).tolist()
        
        results = []
        for cath_id, pred_class, confidence in zip(cath_ids, predicted_classes, confidence_scores):
            result = {
                "cath_id": cath_id,
                "predicted_class_id": int(pred_class),
                "confidence": float(confidence)
            }
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
        
        # Ensure column order: sequence_id, cath_id, predicted_class_id, confidence
        column_order = ["sequence_id", "cath_id", "predicted_class_id", "confidence"]
        results_df = results_df[column_order]
        
        results_df.to_csv(output_path, index=False)
        log.info(f"Predictions saved to: {output_path}")
        
        # Print summary statistics
        log.info("Summary Statistics:")
        log.info(f"  Total sequences: {len(results_df)}")
        log.info(f"  Unique CATH superfamilies: {results_df['cath_id'].nunique()}")
        log.info(f"  Average confidence: {results_df['confidence'].mean():.3f}")
        
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
        "--training_labels", "-t",
        required=True,
        help="Path to training labels CSV file"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="predictions.csv",
        help="Output CSV file for predictions"
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
    
    if not Path(args.training_labels).exists():
        raise FileNotFoundError(f"Training labels file not found: {args.training_labels}")
    
    log.info("🧬 Starting CATHe protein structure classification pipeline")
    
    try:
        # Step 1: Load sequences
        log.info("📁 Loading protein sequences...")
        sequences_data = load_fasta_sequences(args.fasta)
        sequence_ids, sequences = zip(*sequences_data)
        
        # Step 2: Generate embeddings
        log.info("🔄 Generating ProtT5 embeddings...")
        embedder = ProtT5Embedder(device=args.device)
        embeddings = embedder.embed_batch(sequences, batch_size=args.batch_size)
        
        # Step 3: Load classifier
        log.info("🏗️ Loading classifier...")
        classifier = StructureClassifier(
            checkpoint_path=args.checkpoint,
            training_labels_path=args.training_labels
        )
        
        # Step 4: Make predictions
        log.info("🔮 Making CATH superfamily predictions...")
        predictions = classifier.predict_with_cath_labels(embeddings)
        
        # Step 5: Save results
        log.info("💾 Saving predictions...")
        save_predictions(predictions, sequence_ids, args.output)
        
        log.info("✅ Pipeline completed successfully!")
        
    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
