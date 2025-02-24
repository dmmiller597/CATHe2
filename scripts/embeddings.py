import pandas as pd
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
from dataclasses import dataclass
from typing import List
import h5py

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    batch_size: int = 16
    max_length: int = 1024
    normalize: bool = True
    cache_dir: str = "data/embedding_cache"

class ProteinEmbeddingGenerator:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_bfd').to(self.device).half().eval()
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)
        self.valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    def _clean_sequence(self, sequence: str) -> str:
        """Clean and validate protein sequence"""
        if not sequence:
            return ""
        sequence = sequence.upper()
        return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", sequence)

    def generate_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Generate embeddings for protein sequences"""
        # Clean sequences
        sequences = [self._clean_sequence(seq) for seq in sequences if seq]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), self.config.batch_size)):
            batch = sequences[i:i + self.config.batch_size]
            
            # Tokenize
            inputs = self.tokenizer.batch_encode_plus(
                batch,
                add_special_tokens=True,
                padding=True,
                max_length=self.config.max_length,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=inputs['input_ids'], 
                                  attention_mask=inputs['attention_mask'])
                
                # Mean pool and normalize
                mask = inputs['attention_mask'].unsqueeze(-1).expand(
                    outputs.last_hidden_state.size()).half()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                if self.config.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.extend(embeddings.cpu().numpy())

        return np.array(all_embeddings)

    def process_split(self, split_name: str):
        """Process a data split and save embeddings"""
        print(f"Processing {split_name} split...")
        
        # Load sequences
        df = pd.read_parquet(f"data/splits/{split_name}.parquet")
        if split_name == 'train':
            df = df[df['s30_rep']].reset_index(drop=True)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df['sequence'].tolist())
        labels = df['SF'].values
        
        # Save results
        output_dir = Path('data/embeddings')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_dir / f'{split_name}.h5', 'w') as f:
            f.create_dataset('embeddings', data=embeddings, compression='gzip')
            f.create_dataset('labels', data=labels, compression='gzip')
            f.attrs['date'] = str(pd.Timestamp.now())
            f.attrs['model'] = 'Rostlab/prot_t5_xl_bfd'

def main():
    config = EmbeddingConfig()
    generator = ProteinEmbeddingGenerator(config)
    
    for split in ['train', 'val', 'test']:
        generator.process_split(split)

if __name__ == "__main__":
    main()