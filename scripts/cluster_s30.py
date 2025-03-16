#!/usr/bin/env python3
"""Create representative S30 protein dataset using MMseqs2 clustering.

Key functionality:
- Clusters protein sequences at 30% identity threshold
- Identifies representative sequences for each cluster
- Creates filtered dataset of non-redundant proteins
- Maintains original sequence identifiers and metadata

Dependencies: pandas, MMseqs2
"""

import os
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
from typing import Tuple
import logging

def setup_logging() -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_fasta_from_df(df: pd.DataFrame, fasta_path: str) -> None:
    """
    Create a FASTA file from DataFrame sequences.
    
    Args:
        df: DataFrame containing sequence_id and sequence columns
        fasta_path: Path to save the FASTA file
    """
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{row['sequence_id']}\n{row['sequence']}\n")

def run_mmseqs_clustering(
    input_fasta: str,
    output_dir: str,
    seq_id_threshold: float = 0.3
) -> Tuple[str, str]:
    """
    Run MMseqs2 easy-cluster command.
    
    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory for MMseqs2 output
        seq_id_threshold: Sequence identity threshold (default: 0.3 for 30%)
        
    Returns:
        Tuple of paths to representative sequences and cluster TSV
    """
    base_name = os.path.join(output_dir, "clusters")
    
    cmd = [
        "mmseqs", "easy-cluster",
        input_fasta,
        base_name,
        output_dir,
        f"--min-seq-id {seq_id_threshold}",
        "-c 0.8",  # coverage threshold
        "--cov-mode 0",  # coverage mode (0: bidirectional)
        "--threads 8"  # adjust based on available resources
    ]
    
    logging.info("Running MMseqs2 clustering...")
    subprocess.run(" ".join(cmd), shell=True, check=True)
    
    return f"{base_name}_rep_seq.fasta", f"{base_name}_cluster.tsv"

def process_clustering_results(
    original_df: pd.DataFrame,
    cluster_tsv: str
) -> pd.DataFrame:
    """
    Process MMseqs2 clustering results and create filtered DataFrame.
    
    Args:
        original_df: Original DataFrame with sequences
        cluster_tsv: Path to MMseqs2 cluster TSV file
        
    Returns:
        DataFrame containing only representative sequences
    """
    # Read clustering results
    cluster_df = pd.read_csv(cluster_tsv, sep='\t', header=None,
                            names=['representative', 'member'])
    
    # Get representative sequence IDs
    rep_sequences = cluster_df['representative'].unique()
    
    # Filter original DataFrame for representatives
    filtered_df = original_df[original_df['sequence_id'].isin(rep_sequences)].copy()
    
    # Rename H_group column to SF
    filtered_df = filtered_df.rename(columns={'H_group': 'SF'})
    
    # Select only required columns
    filtered_df = filtered_df[['sequence_id', 'sequence', 'SF']]
    
    return filtered_df

def main():
    """Main function to run the clustering pipeline."""
    setup_logging()
    
    # Setup paths
    input_path = Path("data/TED/all_cathe2_s90_reps.parquet")
    output_dir = Path("data/TED/s30")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input data
    logging.info("Reading input parquet file...")
    df = pd.read_parquet(input_path)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create FASTA file
        fasta_path = os.path.join(tmp_dir, "sequences.fasta")
        create_fasta_from_df(df, fasta_path)
        
        # Run clustering
        rep_seqs_file, cluster_tsv = run_mmseqs_clustering(
            fasta_path,
            tmp_dir
        )
        
        # Process results
        logging.info("Processing clustering results...")
        filtered_df = process_clustering_results(df, cluster_tsv)
        
        # Save results
        output_path = output_dir / "s30_representatives.parquet"
        logging.info(f"Saving results to {output_path}")
        filtered_df.to_parquet(output_path, index=False)
        
        logging.info(f"Clustering complete. Found {len(filtered_df)} representative sequences")

if __name__ == "__main__":
    main() 