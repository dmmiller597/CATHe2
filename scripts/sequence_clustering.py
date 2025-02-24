"""
Sequence Clusterer for protein datasets.
Uses MMseqs2 to cluster sequences at various sequence identity thresholds
and adds cluster membership information to the dataset.
"""

import logging
from pathlib import Path
import pandas as pd
import subprocess
from typing import Set
import shutil
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SequenceClusterer:
    """Clusters protein sequences at various sequence identity thresholds using MMseqs2."""
    
    def __init__(self, input_parquet: str, output_dir: str):
        """
        Initialize the SequenceClusterer.
        
        Args:
            input_parquet: Path to input parquet file containing sequence data
            output_dir: Directory where results will be saved
        """
        self.input_path = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create tmp directory for MMseqs2 operations
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
            
    def _run_mmseqs_cluster(self, input_fasta: Path, seq_id: float) -> Path:
        """
        Run MMseqs2 clustering at specified sequence identity threshold.
        
        Args:
            input_fasta: Path to input FASTA file
            seq_id: Sequence identity threshold (0.0-1.0)
            
        Returns:
            Path to cluster results file
        """
        cluster_dir = self.tmp_dir / f"s{int(seq_id*100)}"
        cluster_dir.mkdir(exist_ok=True)
        
        cluster_prefix = cluster_dir / "clusters"
        result_file = cluster_dir / "cluster.tsv"
        
        cmd = [
            "mmseqs", "easy-cluster",
            str(input_fasta),
            str(cluster_prefix),
            str(cluster_dir),
            "--min-seq-id", str(seq_id),
            "-c", "0.8",  # coverage threshold
            "--cov-mode", "1",  # coverage mode (1 = shorter sequence)
        ]
        
        try:
            logger.info(f"Running clustering at {seq_id:.2f} sequence identity")
            subprocess.run(cmd, check=True, text=True)
            
            # MMseqs2 easy-cluster outputs results to cluster_prefix_cluster.tsv
            result_path = Path(f"{cluster_prefix}_cluster.tsv")
            if result_path.exists():
                return result_path
            else:
                raise FileNotFoundError(f"Expected cluster results at {result_path}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Clustering failed: {e.stderr}")
            raise
            
    def _process_cluster_results(self, cluster_file: Path) -> Set[str]:
        """Process cluster results to get representative sequence IDs."""
        # MMseqs2 cluster output format: RepresentativeID\tMemberID
        with open(cluster_file) as f:
            return {line.strip().split('\t')[0] for line in f}
            
    def cluster_sequences(self, thresholds: list[float] = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]):
        """
        Cluster sequences at multiple identity thresholds and add membership columns.
        
        Args:
            thresholds: List of sequence identity thresholds to use for clustering
        """
        try:
            # Save sequences to FASTA
            input_fasta = self.tmp_dir / "input.fasta"
            self._save_fasta(input_fasta)
            
            # Cluster at each threshold and collect representatives
            for seq_id in thresholds:
                cluster_results = self._run_mmseqs_cluster(input_fasta, seq_id)
                representatives = self._process_cluster_results(cluster_results)
                
                # Add binary column for cluster membership
                col_name = f"s{int(seq_id*100)}_rep"
                self.df[col_name] = self.df['sequence_id'].isin(representatives)
                
                logger.info(
                    f"Identified {len(representatives):,} representatives "
                    f"at {seq_id:.2f} sequence identity "
                    f"({len(representatives)/len(self.df):.1%} of sequences)"
                )
            
            # Save enhanced dataset
            output_path = self.output_dir / self.input_path.name
            self.df.to_parquet(output_path)
            logger.info(f"Saved enhanced dataset to {output_path}")
            
        finally:
            # Cleanup
            if self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)
                
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
        Cluster protein sequences using MMseqs2 at various sequence identity thresholds.
        Input file must be a Parquet file containing at minimum these columns:
        - sequence_id (str): Unique identifier for each sequence
        - sequence (str): Protein sequence in single-letter amino acid code
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input Parquet file containing sequence data'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Directory where results will be saved'
    )
    parser.add_argument(
        '-t', '--thresholds',
        type=float,
        nargs='+',
        default=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        help='Sequence identity thresholds (between 0 and 1) for clustering'
    )
    return parser.parse_args()

def main():
    """Main entry point with error handling."""
    try:
        args = parse_args()

        clusterer = SequenceClusterer(
            input_parquet=args.input,
            output_dir=args.output_dir
        )
        clusterer.cluster_sequences(thresholds=args.thresholds)
        logger.info("Sequence clustering completed successfully")
        
    except Exception as e:
        logger.error(f"Sequence clustering failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()