"""
Sequence Clusterer for protein datasets.
Uses MMseqs2 to cluster sequences at 30% sequence identity threshold.
"""

import logging
from pathlib import Path
import pandas as pd
import subprocess
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SequenceClusterer:
    """Clusters protein sequences at 30% sequence identity using MMseqs2."""
    
    def __init__(self, input_parquet: str, output_dir: str):
        """Initialize the SequenceClusterer."""
        self.input_path = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.tmp_dir = Path("tmp/clustering")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Loading sequences from {input_parquet}")
        self.df = pd.read_parquet(input_parquet)
        
    def _save_fasta(self, filename: Path) -> None:
        """Save sequences to FASTA format."""
        logger.debug(f"Saving {len(self.df):,} sequences to FASTA: {filename}")
        fasta_entries = '>' + self.df['sequence_id'] + '\n' + self.df['sequence'] + '\n'
        with open(filename, 'w') as f:
            f.writelines(fasta_entries)
            
    def _run_mmseqs_cluster(self, input_fasta: Path) -> Path:
        """Run MMseqs2 clustering with standard settings."""
        cluster_dir = self.tmp_dir / "s30"
        cluster_dir.mkdir(exist_ok=True)
        
        cluster_prefix = cluster_dir / "clusters"
        
        cmd = [
            "mmseqs", "easy-cluster",
            str(input_fasta),
            str(cluster_prefix),
            str(cluster_dir),
            "--min-seq-id", "0.3",
            "-c", "0.8",  # coverage threshold
            "--cov-mode", "1",  # coverage mode (1 = shorter sequence)
        ]
        
        try:
            logger.info("Running clustering at 0.30 sequence identity")
            subprocess.run(cmd, check=True, text=True)
            
            result_path = Path(f"{cluster_prefix}_cluster.tsv")
            if not result_path.exists():
                raise FileNotFoundError(f"Expected cluster results at {result_path}")
            return result_path
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Clustering failed: {e.stderr}")
            raise
            
    def cluster_sequences(self):
        """Cluster sequences and add s30_rep column."""
        try:
            input_fasta = self.tmp_dir / "input.fasta"
            self._save_fasta(input_fasta)
            
            cluster_results = self._run_mmseqs_cluster(input_fasta)
            
            # Process results
            with open(cluster_results) as f:
                representatives = {line.strip().split('\t')[0] for line in f}
            
            # Add binary column for cluster membership
            self.df['s30_rep'] = self.df['sequence_id'].isin(representatives)
            
            logger.info(
                f"Identified {len(representatives):,} representatives "
                f"({len(representatives)/len(self.df):.1%} of sequences)"
            )
            
            # Save enhanced dataset
            output_path = self.output_dir / self.input_path.name
            self.df.to_parquet(output_path)
            logger.info(f"Saved enhanced dataset to {output_path}")
            
        finally:
            if self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)

def main():
    """Main entry point with error handling."""
    try:
        clusterer = SequenceClusterer(
            input_parquet="data/TED/all_cathe2_s90_reps.parquet",
            output_dir="data/TED/s30/"
        )
        clusterer.cluster_sequences()
        logger.info("Sequence clustering completed successfully")
        
    except Exception as e:
        logger.error(f"Sequence clustering failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()