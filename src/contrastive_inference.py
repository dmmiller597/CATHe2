import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, List, Dict, Any
from sklearn.neighbors import NearestNeighbors
from rich.progress import Progress
from models.contrasted import ContrastiveCATHeModel
from utils import get_logger

log = get_logger()

class ContrastiveInference:
    """Class for performing inference with a trained contrastive model."""
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        reference_embeddings: Union[str, Path],
        reference_labels: Union[str, Path],
        k: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize contrastive inference.
        
        Args:
            model_path: Path to the trained model checkpoint
            reference_embeddings: Path to reference embeddings file (NPZ)
            reference_labels: Path to reference labels file (CSV)
            k: Number of nearest neighbors to consider
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.k = k
        
        # Load the model
        log.info(f"Loading model from {model_path}")
        self.model = ContrastiveCATHeModel.load_from_checkpoint(
            model_path,
            map_location=device
        )
        self.model.eval()
        self.model.to(device)
        
        # Load reference data
        log.info(f"Loading reference embeddings from {reference_embeddings}")
        ref_data = np.load(reference_embeddings)
        if 'arr_0' in ref_data:
            raw_embeddings = ref_data['arr_0']
        else:
            raw_embeddings = ref_data['embeddings']
            
        # Load labels
        log.info(f"Loading reference labels from {reference_labels}")
        labels_df = pd.read_csv(reference_labels)
        self.ref_labels = labels_df['SF'].values
        
        # Project reference embeddings
        log.info(f"Projecting {len(raw_embeddings)} reference embeddings")
        self.ref_embeddings = self._project_embeddings(raw_embeddings)
        
        # Create nearest neighbor index
        log.info(f"Building nearest neighbor index with k={k}")
        self.nn_index = NearestNeighbors(n_neighbors=k)
        self.nn_index.fit(self.ref_embeddings)
    
    def _project_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings using the trained model.
        
        Args:
            embeddings: Raw protein embeddings
            
        Returns:
            Projected embeddings
        """
        batch_size = 256  # Large batch size for faster processing
        projected = []
        
        with torch.no_grad(), Progress() as progress:
            task = progress.add_task("Projecting embeddings...", total=len(embeddings))
            
            for i in range(0, len(embeddings), batch_size):
                batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(self.device)
                proj = self.model(batch).cpu().numpy()
                projected.append(proj)
                progress.update(task, advance=len(batch))
                
        return np.vstack(projected)
    
    def predict(
        self, 
        query_embeddings: Union[np.ndarray, str, Path]
    ) -> Tuple[List[str], List[float]]:
        """Predict CATH superfamily for query embeddings.
        
        Args:
            query_embeddings: Query protein embeddings or path to embeddings file
            
        Returns:
            Tuple of (predicted superfamilies, confidence scores)
        """
        # Load embeddings if path is provided
        if isinstance(query_embeddings, (str, Path)):
            query_data = np.load(query_embeddings)
            if 'arr_0' in query_data:
                query_embeddings = query_data['arr_0']
            else:
                query_embeddings = query_data['embeddings']
        
        # Project query embeddings
        log.info(f"Projecting {len(query_embeddings)} query embeddings")
        query_projected = self._project_embeddings(query_embeddings)
        
        # Find nearest neighbors
        log.info("Finding nearest neighbors")
        distances, indices = self.nn_index.kneighbors(query_projected)
        
        # Get predictions
        predictions = []
        confidences = []
        
        for i in range(len(query_projected)):
            # Get neighbor labels
            neighbor_labels = [self.ref_labels[idx] for idx in indices[i]]
            
            # Count label frequencies
            label_counts = {}
            for j, label in enumerate(neighbor_labels):
                if label not in label_counts:
                    label_counts[label] = 0
                # Weight by inverse distance (closer neighbors have more influence)
                weight = 1.0 / (distances[i, j] + 1e-8)
                label_counts[label] += weight
            
            # Get most frequent label and its confidence
            predicted_label = max(label_counts.items(), key=lambda x: x[1])[0]
            total_weight = sum(label_counts.values())
            confidence = label_counts[predicted_label] / total_weight
            
            predictions.append(predicted_label)
            confidences.append(confidence)
        
        return predictions, confidences 