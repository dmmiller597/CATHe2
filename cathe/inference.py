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
        device: str = "auto",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False
    ) -> None:
        """Initialize ProtT5 model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier or local path to ProtT5
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            cache_dir: Custom cache directory for model downloads
            local_files_only: If True, only use locally cached files
        """
        self.device = self._get_device(device)
        log.info(f"Loading ProtT5 model on {self.device}")
        
        # Set up cache directory if provided
        if cache_dir:
            cache_path = Path(cache_dir).resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            log.info(f"Using custom cache directory: {cache_path}")
        else:
            cache_path = None
        
        try:
            # Check if it's a local path
            if Path(model_name).exists():
                log.info(f"Loading model from local path: {model_name}")
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
                self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
            else:
                # Try to load from HuggingFace Hub
                log.info(f"Loading model from HuggingFace Hub: {model_name}")
                self.tokenizer = T5Tokenizer.from_pretrained(
                    model_name, 
                    do_lower_case=False,
                    cache_dir=cache_path,
                    local_files_only=local_files_only
                )
                self.model = T5EncoderModel.from_pretrained(
                    model_name,
                    cache_dir=cache_path,
                    local_files_only=local_files_only
                ).to(self.device)
            
            self.model.eval()
            log.info("ProtT5 model loaded successfully")
            
        except OSError as e:
            if "Disk quota exceeded" in str(e):
                log.error("Disk quota exceeded while downloading model. Solutions:")
                log.error("1. Set a custom cache directory with --cache_dir")
                log.error("2. Use --local_files_only if model is already cached")
                log.error("3. Use a smaller model variant")
                log.error("4. Clear space in your home directory")
            raise
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

class CATHeLabelEncoder:
    """Manages CATH superfamily label encoding exactly as done during training."""
    
    def __init__(self, model_num_classes: int):
        """Initialize with expected number of classes from model.
        
        Args:
            model_num_classes: Number of classes the model was trained on
        """
        self.model_num_classes = model_num_classes
        self.label_encoder: Optional[LabelEncoder] = None
    
    def create_from_training_labels(self, training_labels_path: str) -> LabelEncoder:
        """Create label encoder exactly as done during training.
        
        This replicates the exact process from data_module.py:
        1. Read training labels CSV with 'SF' column
        2. Fit LabelEncoder on df['SF'] 
        3. LabelEncoder automatically sorts classes alphabetically
        
        Args:
            training_labels_path: Path to training labels CSV file
            
        Returns:
            LabelEncoder fitted exactly as during training
        """
        log.info(f"Creating label encoder from training data: {training_labels_path}")
        
        # Replicate exact training process from data_module.py
        df = pd.read_csv(training_labels_path)
        if 'SF' not in df.columns:
            raise ValueError(f"Training labels CSV must contain 'SF' column. Found: {df.columns.tolist()}")
        
        # Fit LabelEncoder exactly as in training (data_module.py line 83)
        self.label_encoder = LabelEncoder().fit(df['SF'])
        
        log.info(f"Label encoder created with {len(self.label_encoder.classes_)} classes")
        log.info(f"First 10 classes: {list(self.label_encoder.classes_[:10])}")
        
        self._validate_against_model()
        return self.label_encoder
    
    def load_from_file(self, label_encoder_path: str) -> LabelEncoder:
        """Load label encoder from pickle file and validate.
        
        Args:
            label_encoder_path: Path to saved label encoder pickle file
            
        Returns:
            Validated label encoder
        """
        log.info(f"Loading label encoder from: {label_encoder_path}")
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        log.info(f"Loaded label encoder with {len(self.label_encoder.classes_)} classes")
        
        self._validate_against_model()
        return self.label_encoder
    
    def get_or_create_encoder(
        self, 
        label_encoder_path: Optional[str] = None,
        training_labels_path: Optional[str] = None,
        auto_save: bool = True
    ) -> LabelEncoder:
        """Get label encoder using the standard workflow.
        
        Priority order:
        1. Load from saved pickle file if exists
        2. Create from training labels if available  
        3. Fail with informative error
        
        Args:
            label_encoder_path: Path to saved label encoder pickle file
            training_labels_path: Path to training labels CSV file
            auto_save: Whether to save newly created encoder
            
        Returns:
            Validated label encoder
        """
        # Try to load existing encoder first
        if label_encoder_path and Path(label_encoder_path).exists():
            try:
                return self.load_from_file(label_encoder_path)
            except Exception as e:
                log.warning(f"Failed to load encoder from {label_encoder_path}: {e}")
        
        # Try to create from training data
        if training_labels_path and Path(training_labels_path).exists():
            encoder = self.create_from_training_labels(training_labels_path)
            
            # Auto-save if requested and path provided
            if auto_save and label_encoder_path:
                self.save_encoder(encoder, label_encoder_path)
            
            return encoder
        
        # If all fails, provide helpful error message
        raise ValueError(
            "Cannot create label encoder. Please provide either:\n"
            f"1. Existing encoder file: {label_encoder_path}\n" 
            f"2. Training labels CSV: {training_labels_path}\n"
            "The encoder is required to map model predictions to CATH superfamily IDs."
        )
    
    def _validate_against_model(self) -> None:
        """Validate encoder matches model's expected number of classes."""
        if self.label_encoder is None:
            raise RuntimeError("No label encoder loaded")
        
        encoder_classes = len(self.label_encoder.classes_)
        if encoder_classes != self.model_num_classes:
            raise ValueError(
                f"Label encoder mismatch!\n"
                f"  Model expects: {self.model_num_classes} classes\n"
                f"  Encoder has: {encoder_classes} classes\n"
                f"  Model and encoder must be from the same training run."
            )
        
        log.info(f"✓ Label encoder validation passed: {encoder_classes} classes match model")
    
    def save_encoder(self, encoder: LabelEncoder, output_path: str) -> None:
        """Save label encoder to pickle file.
        
        Args:
            encoder: Label encoder to save
            output_path: Path to save pickle file
        """
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(encoder, f)
            log.info(f"Label encoder saved to: {output_path}")
        except Exception as e:
            log.warning(f"Failed to save encoder: {e}")
    
    def decode_predictions(self, class_ids: np.ndarray) -> List[str]:
        """Decode integer class IDs to CATH superfamily strings.
        
        Args:
            class_ids: Array of predicted class integers
            
        Returns:
            List of CATH superfamily IDs (e.g., ['3.30.450.10', '1.10.8.10', ...])
        """
        if self.label_encoder is None:
            raise RuntimeError("No label encoder available for decoding")
        
        try:
            return self.label_encoder.inverse_transform(class_ids).tolist()
        except Exception as e:
            log.error(f"Failed to decode class IDs: {e}")
            # Fallback to generic names
            return [f"Unknown_Class_{cid}" for cid in class_ids]

class StructureClassifier:
    """CATH superfamily classifier using trained CATHe model."""
    
    def __init__(
        self, 
        checkpoint_path: str, 
        label_encoder_path: Optional[str] = None,
        training_labels_path: Optional[str] = None
    ) -> None:
        """Initialize classifier with trained model and label encoder.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            label_encoder_path: Path to saved label encoder pickle file
            training_labels_path: Path to training labels CSV (for creating encoder)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model first to get number of classes
        try:
            log.info(f"Loading model from: {checkpoint_path}")
            self.model = CATHeClassifier.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            self.model.to(self.device)
            
            # Extract number of classes from model
            self.num_classes = self.model.num_classes
            log.info(f"Model expects {self.num_classes} classes")
            
        except Exception as e:
            log.error(f"Failed to load model checkpoint: {e}")
            raise
        
        # Initialize label encoder with exact training scheme
        self.label_manager = CATHeLabelEncoder(self.num_classes)
        self.label_encoder = self.label_manager.get_or_create_encoder(
            label_encoder_path=label_encoder_path,
            training_labels_path=training_labels_path,
            auto_save=True
        )
        
        log.info(f"✓ Classifier ready with {self.num_classes} CATH superfamilies")
    
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
        
        # Decode to CATH superfamily IDs using exact training scheme
        cath_ids = self.label_manager.decode_predictions(predicted_classes)
        
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
    """Save predictions to CSV file in the required format.
    
    Args:
        predictions: List of prediction dictionaries
        sequence_ids: List of sequence identifiers  
        output_path: Path to save results
    """
    try:
        results_df = pd.DataFrame(predictions)
        results_df.insert(0, "sequence_id", sequence_ids)
        
        # Ensure exact column order: sequence_id, cath_id, predicted_class_id, confidence
        column_order = ["sequence_id", "cath_id", "predicted_class_id", "confidence"]
        results_df = results_df[column_order]
        
        results_df.to_csv(output_path, index=False)
        log.info(f"Predictions saved to: {output_path}")
        
        # Print summary statistics
        log.info("Summary Statistics:")
        log.info(f"  Total sequences: {len(results_df)}")
        log.info(f"  Unique CATH superfamilies: {results_df['cath_id'].nunique()}")
        log.info(f"  Average confidence: {results_df['confidence'].mean():.3f}")
        log.info(f"  Confidence range: [{results_df['confidence'].min():.3f}, {results_df['confidence'].max():.3f}]")
        
        # Show most common predictions
        top_predictions = results_df['cath_id'].value_counts().head(5)
        log.info("Top 5 predicted superfamilies:")
        for cath_id, count in top_predictions.items():
            log.info(f"  {cath_id}: {count} sequences")
        
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
        "--training_labels",
        help="Path to training labels CSV file (for creating label encoder)"
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
    
    parser.add_argument(
        "--model_name",
        default="Rostlab/prot_t5_xl_half_uniref50-enc",
        help="ProtT5 model name or local path"
    )
    
    parser.add_argument(
        "--cache_dir",
        help="Custom cache directory for model downloads"
    )
    
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use locally cached model files"
    )
    
    return parser

def main() -> None:
    """Main inference pipeline with exact training label encoding."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.fasta).exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    log.info("🧬 Starting CATHe protein structure classification pipeline")
    
    try:
        # Step 1: Load sequences
        log.info("📁 Loading protein sequences...")
        sequences_data = load_fasta_sequences(args.fasta)
        sequence_ids, sequences = zip(*sequences_data)
        
        # Step 2: Generate embeddings
        log.info("🔄 Generating ProtT5 embeddings...")
        embedder = ProtT5Embedder(
            model_name=args.model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only
        )
        embeddings = embedder.embed_batch(sequences, batch_size=args.batch_size)
        
        # Step 3: Load classifier with exact training label encoding
        log.info("🏗️ Loading classifier with training-compatible label encoding...")
        classifier = StructureClassifier(
            checkpoint_path=args.checkpoint,
            label_encoder_path=args.label_encoder,
            training_labels_path=args.training_labels
        )
        
        # Step 4: Make predictions with CATH labels
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
