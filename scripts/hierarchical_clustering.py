"""
Hierarchical Sequence Clusterer for protein datasets.
Uses MMseqs2 to cluster sequences in a hierarchical manner from high to low identity thresholds.
Starting with s80 representatives, then clustering those at s70, s60, etc. down to s30.
"""

import logging
from pathlib import Path
import pandas as pd
import subprocess
import shutil
import os
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HierarchicalClusterer:
    """
    Clusters protein sequences hierarchically at multiple sequence identity thresholds.
    Starting with high identity threshold, only representatives are used for the next threshold.
    """
    
    def __init__(self, input_parquet: str, output_dir: str, thresholds=None, threads=None):
        """Initialize the HierarchicalClusterer.
        
        Args:
            input_parquet: Path to input parquet file containing sequences
            output_dir: Directory to save results
            thresholds: List of sequence identity thresholds (default: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            threads: Number of threads to use (default: automatically determined based on CPU count)
        """
        self.input_path = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.tmp_dir = Path("/state/partition1/scratch0/tmp/")
        
        # Define identity thresholds (from high to low)
        self.thresholds = thresholds or [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        
        # Determine number of threads
        available_cores = multiprocessing.cpu_count()
        # Use all cores minus 1 by default, to leave one core for the system
        # But ensure at least 1 thread is used
        self.threads = threads or max(1, available_cores - 1)
        logger.info(f"Using {self.threads} threads (of {available_cores} available cores)")
        
        logger.info(f"Loading sequences from {input_parquet}")
        self.df = pd.read_parquet(input_parquet)
        
        # Create directory for temporary files
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        
    def _save_fasta(self, df, filename: Path) -> None:
        """Save sequences to FASTA format.
        
        Args:
            df: DataFrame containing sequence_id and sequence columns
            filename: Path to output FASTA file
        """
        logger.debug(f"Saving {len(df):,} sequences to FASTA: {filename}")
        fasta_entries = '>' + df['sequence_id'] + '\n' + df['sequence'] + '\n'
        with open(filename, 'w') as f:
            f.writelines(fasta_entries)
            
    def _run_mmseqs_cluster(self, input_fasta: Path, identity: float) -> Path:
        """Run MMseqs2 clustering at specified sequence identity.
        
        Args:
            input_fasta: Path to input FASTA file
            identity: Sequence identity threshold (0.0-1.0)
            
        Returns:
            Path to the clustering results file
        """
        # Format the identity for directory naming
        id_str = f"s{int(identity * 100)}"
        cluster_dir = self.tmp_dir / id_str
        cluster_dir.mkdir(exist_ok=True)
        
        cluster_prefix = cluster_dir / "clusters"
        
        cmd = [
            "mmseqs", "easy-cluster",
            str(input_fasta),
            str(cluster_prefix),
            str(cluster_dir),
            "--min-seq-id", str(identity),
            "-c", "0.8",  # coverage threshold
            "--cov-mode", "2",  # coverage mode (2 = query sequence)
            "--threads", str(self.threads),
            "--cluster-mode", "0"  # 0 = centroid as representative (most connected sequence)
        ]
        
        try:
            logger.info(f"Running clustering at {identity:.2f} sequence identity with {self.threads} threads")
            subprocess.run(cmd, check=True, text=True)
            
            result_path = Path(f"{cluster_prefix}_cluster.tsv")
            if not result_path.exists():
                raise FileNotFoundError(f"Expected cluster results at {result_path}")
            return result_path
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Clustering failed: {e.stderr}")
            raise
            
    def cluster_sequences(self):
        """Perform hierarchical clustering at multiple sequence identity thresholds."""
        try:
            current_df = self.df.copy()
            
            # Sort thresholds from high to low to ensure proper hierarchical clustering
            sorted_thresholds = sorted(self.thresholds, reverse=True)
            
            for identity in sorted_thresholds:
                id_str = f"s{int(identity * 100)}"
                rep_col = f"{id_str}_rep"
                
                input_fasta = self.tmp_dir / f"{id_str}_input.fasta"
                self._save_fasta(current_df, input_fasta)
                
                cluster_results = self._run_mmseqs_cluster(input_fasta, identity)
                
                # Process results
                with open(cluster_results) as f:
                    representatives = {line.strip().split('\t')[0] for line in f}
                
                # Add binary column for cluster membership to the original dataframe
                self.df[rep_col] = False
                self.df.loc[self.df['sequence_id'].isin(representatives), rep_col] = True
                
                logger.info(
                    f"Identified {len(representatives):,} {id_str} representatives "
                    f"({len(representatives)/len(current_df):.1%} of input sequences)"
                )
                
                # Filter to only use representatives for the next clustering level
                current_df = current_df[current_df['sequence_id'].isin(representatives)].copy()
            
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
        clusterer = HierarchicalClusterer(
            input_parquet="data/TED/all_reps.parquet",
            output_dir="data/TED/hierarchical_clusters",
            thresholds=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
            # threads parameter automatically determined if not specified
        )
        clusterer.cluster_sequences()
        logger.info("Hierarchical sequence clustering completed successfully")
        
    except Exception as e:
        logger.error(f"Hierarchical sequence clustering failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()