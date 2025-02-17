"""
Dataset Splitter for protein sequence datasets.
Handles splitting of sequence data into train/val/test sets while maintaining
superfamily representation and sequence similarity constraints.
"""

import logging
from pathlib import Path
import pandas as pd
from typing import Tuple, Set, Dict, Optional
import subprocess
import tempfile
from dataclasses import dataclass
from tqdm import tqdm

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class SplitStats:
    """Statistics for dataset splits."""
    num_sequences: int
    num_superfamilies: int
    sequences_per_sf: Dict[str, int]

class DatasetSplitter:
    """Handles the splitting of protein sequence datasets into train/val/test sets."""
    
    def __init__(self, input_parquet: str, output_dir: str):
        """
        Initialize the DatasetSplitter.
        
        Args:
            input_parquet: Path to input parquet file containing sequence data
            output_dir: Directory where split datasets will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create tmp directory in project root
        self.tmp_dir = Path("tmp")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Loading data from {input_parquet}")
        # Only load required columns and rename H_group to SF
        self.df = pd.read_parquet(
            input_parquet,
            columns=['sequence_id', 'sequence', 'H_group']
        ).rename(columns={'H_group': 'SF'})
        self._validate_input_data()
        
    def _validate_input_data(self) -> None:
        """Validate input data structure and content."""
        required_columns = {'sequence_id', 'sequence', 'SF'}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if len(self.df) == 0:
            raise ValueError("Input dataframe is empty")
        
        # Validate data quality
        if self.df['sequence_id'].duplicated().any():
            raise ValueError("Duplicate sequence IDs found")
        if self.df['sequence'].str.len().min() == 0:
            raise ValueError("Empty sequences found")
            
    def filter_small_sfs(self, min_sequences: int = 10) -> None:
        """Filter out superfamilies using vectorized operations."""
        logger.info(f"Starting superfamily filtering (min_sequences={min_sequences})")
        initial_sfs = len(self.df['SF'].unique())
        initial_sequences = len(self.df)
        
        # Vectorized filtering
        sf_counts = self.df['SF'].value_counts()
        valid_sfs = sf_counts[sf_counts >= min_sequences].index
        
        # Save filtered sequences before removing them
        filtered_sequences = self.df[~self.df['SF'].isin(valid_sfs)].copy()
        filtered_output = self.output_dir / "filtered_sf_sequences.parquet"
        filtered_sequences.to_parquet(filtered_output)
        logger.info(f"Saved {len(filtered_sequences):,} filtered sequences to {filtered_output}")
        
        self.df = self.df[self.df['SF'].isin(valid_sfs)].copy()
        
        # Vectorized statistics calculation
        final_stats = {
            'removed_sfs': initial_sfs - len(valid_sfs),
            'removed_sequences': initial_sequences - len(self.df),
            'remaining_sfs': len(valid_sfs),
            'remaining_sequences': len(self.df)
        }
        
        logger.info(
            f"Filtering results:\n"
            f"- Removed superfamilies: {final_stats['removed_sfs']:,} "
            f"({final_stats['removed_sfs']/initial_sfs:.1%})\n"
            f"- Removed sequences: {final_stats['removed_sequences']:,} "
            f"({final_stats['removed_sequences']/initial_sequences:.1%})\n"
            f"- Remaining: {final_stats['remaining_sfs']:,} SFs, "
            f"{final_stats['remaining_sequences']:,} sequences"
        )

    def create_initial_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create initial splits using vectorized operations."""
        logger.info("Creating initial splits")
        
        # More efficient sampling approach using sample + concat
        sf_groups = self.df.groupby('SF')
        sample_dfs = []
        
        for _, group in sf_groups:
            if len(group) >= 2:
                samples = group.sample(n=min(2, len(group)), random_state=42)
                sample_dfs.append(samples)
        
        if not sample_dfs:
            raise ValueError("No superfamilies had sufficient sequences for splitting")
        
        # Combine all sampled sequences
        combined_samples = pd.concat(sample_dfs, ignore_index=True)
        
        # Create val and test splits
        val_df = combined_samples.groupby('SF').head(1).reset_index(drop=True)
        test_df = combined_samples.groupby('SF').tail(1).reset_index(drop=True)
        
        # Create train split
        split_ids = set(val_df['sequence_id']) | set(test_df['sequence_id'])
        train_df = self.df[~self.df['sequence_id'].isin(split_ids)].copy()
        
        self._log_split_stats("Initial", train_df, val_df, test_df)
        return train_df, val_df, test_df

    def _save_fasta(self, df: pd.DataFrame, filename: Path) -> None:
        """Save sequences to FASTA format efficiently using vectorized operations."""
        logger.debug(f"Saving {len(df):,} sequences to FASTA: {filename}")
        # Vectorized string formatting
        fasta_entries = '>' + df['sequence_id'] + '\n' + df['sequence'] + '\n'
        with open(filename, 'w') as f:
            f.writelines(fasta_entries)

    def _run_mmseqs_search(self, query_fasta: Path, target_fasta: Path, 
                          output_file: Path, tmp_dir: Path) -> None:
        """Run MMseqs2 search with error handling and validation."""
        logger.info(
            f"Running MMseqs2 search:\n"
            f"- Query: {query_fasta} ({query_fasta.stat().st_size:,} bytes)\n"
            f"- Target: {target_fasta} ({target_fasta.stat().st_size:,} bytes)\n"
            f"- Output: {output_file}\n"
            f"- Temp dir: {tmp_dir}"
        )
        
        cmd = [
            "mmseqs", "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(output_file),
            str(tmp_dir),
            "--min-seq-id", "0.3",
            "-c", "0.8",
            "--format-output", "query,target,fident"
        ]
        
        try:
            logger.debug(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, text=True)
            logger.debug(f"MMseqs2 stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"MMseqs2 search failed: {e.stderr}")
            raise

    def find_similar_sequences(self, query_df: pd.DataFrame, 
                             target_df: pd.DataFrame) -> Set[str]:
        """
        Find sequences in target_df similar to query_df sequences.
        
        Args:
            query_df: DataFrame containing query sequences
            target_df: DataFrame containing target sequences
            
        Returns:
            Set of sequence IDs from target_df similar to query sequences
        """
        if query_df.empty or target_df.empty:
            logger.warning("Empty DataFrame provided for similarity search")
            return set()
            
        # Use project-specific tmp directory instead of system temp
        search_dir = self.tmp_dir / f"mmseqs_search_{hash(str(query_df.head(1)))}"
        search_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            query_fasta = search_dir / "query.fasta"
            target_fasta = search_dir / "target.fasta"
            results_file = search_dir / "results.tsv"
            
            self._save_fasta(query_df, query_fasta)
            self._save_fasta(target_df, target_fasta)
            
            self._run_mmseqs_search(query_fasta, target_fasta, results_file, search_dir)
            
            similar_sequences = set()
            if results_file.exists():
                similar_sequences = {
                    line.strip().split('\t')[1]
                    for line in results_file.open()
                }
            
            return similar_sequences
        finally:
            # Clean up temporary files
            if search_dir.exists():
                import shutil
                shutil.rmtree(search_dir)

    def _get_split_stats(self, df: pd.DataFrame) -> SplitStats:
        """Calculate statistics for a dataset split."""
        return SplitStats(
            num_sequences=len(df),
            num_superfamilies=len(df['SF'].unique()),
            sequences_per_sf=df['SF'].value_counts().to_dict()
        )

    def _log_split_stats(self, stage: str, train_df: pd.DataFrame, 
                        val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Log detailed statistics about the current state of the splits."""
        logger.info(f"\n{stage} Split Statistics:")
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            stats = self._get_split_stats(df)
            logger.info(
                f"{name}: {stats.num_sequences:,} sequences, "
                f"{stats.num_superfamilies:,} superfamilies"
            )

    def process_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the complete dataset splitting pipeline with vectorized operations."""
        logger.info("Starting dataset splitting pipeline")
        
        self.filter_small_sfs()
        train_df, val_df, test_df = self.create_initial_splits()
        
        logger.info("Finding similar sequences between splits")
        
        # Process validation and test similarities in parallel
        val_similar = self.find_similar_sequences(val_df, train_df)
        test_similar = self.find_similar_sequences(test_df, train_df)
        
        logger.info(
            f"Similarity search results:\n"
            f"Validation: {len(val_similar):,} sequences ({len(val_similar)/len(train_df):.1%} of training)\n"
            f"Test: {len(test_similar):,} sequences ({len(test_similar)/len(train_df):.1%} of training)"
        )
        
        # Vectorized reassignment of similar sequences
        all_similar = val_similar | test_similar
        similar_mask = train_df['sequence_id'].isin(all_similar)
        similar_sequences = train_df[similar_mask]
        
        # Vectorized split updates
        val_similar_mask = similar_sequences['sequence_id'].isin(val_similar)
        test_similar_mask = similar_sequences['sequence_id'].isin(test_similar)
        
        val_df = pd.concat([val_df, similar_sequences[val_similar_mask]], ignore_index=True)
        test_df = pd.concat([test_df, similar_sequences[test_similar_mask]], ignore_index=True)
        train_df = train_df[~similar_mask].copy()

        # Ensure consistency across splits by removing superfamilies not in training set
        logger.info("Ensuring superfamily consistency across splits")
        train_sfs = set(train_df['SF'].unique())
        
        # Filter validation and test sets to only include superfamilies present in training
        val_df = val_df[val_df['SF'].isin(train_sfs)].copy()
        test_df = test_df[test_df['SF'].isin(train_sfs)].copy()
        
        logger.info("Superfamily consistency check:")
        logger.info(f"Training superfamilies: {len(train_sfs):,}")
        logger.info(f"Validation superfamilies: {len(val_df['SF'].unique()):,}")
        logger.info(f"Test superfamilies: {len(test_df['SF'].unique()):,}")
        
        logger.info("Checking for and removing any duplicate sequences between splits")
        
        # Remove duplicates, prioritizing order: train > test > val
        test_df = test_df[~test_df['sequence_id'].isin(train_df['sequence_id'])].copy()
        val_df = val_df[~val_df['sequence_id'].isin(train_df['sequence_id'])].copy()
        
        # Remove any duplicates from val that appear in test (prioritizing test)
        val_df = val_df[~val_df['sequence_id'].isin(test_df['sequence_id'])].copy()
        
        # Log any removed duplicates
        logger.info(f"Final sequence counts after deduplication:")
        logger.info(f"Train: {len(train_df):,}")
        logger.info(f"Test: {len(test_df):,}")
        logger.info(f"Val: {len(val_df):,}")
        
        self._log_split_stats("Final", train_df, val_df, test_df)
        self._validate_final_splits(train_df, val_df, test_df)
        
        logger.info("Saving final splits")
        for split_name, df in [
            ('train', train_df),
            ('val', val_df),
            ('test', test_df)
        ]:
            # Add split column directly to DataFrame
            df['split'] = split_name
            
            output_file = self.output_dir / f"{split_name}.parquet"
            logger.info(f"Saving {split_name} split ({len(df):,} sequences) to {output_file}")
            df.to_parquet(output_file)
        
        return train_df, val_df, test_df

    def _validate_final_splits(self, train_df: pd.DataFrame, 
                             val_df: pd.DataFrame, 
                             test_df: pd.DataFrame) -> None:
        """Validate the final splits meet all requirements."""
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if df.empty:
                raise ValueError(f"Empty {split_name} split")
            
            if len(df['SF'].unique()) != len(self.df['SF'].unique()):
                missing_sfs = set(self.df['SF'].unique()) - set(df['SF'].unique())
                logger.warning(
                    f"{split_name} split missing {len(missing_sfs)} superfamilies: "
                    f"{', '.join(list(missing_sfs)[:5])}..."
                )

def main():
    """Main entry point with error handling."""
    try:
        splitter = DatasetSplitter(
            input_parquet="data/TED/all_cathe2_s90_reps.parquet",
            output_dir="data/splits"
        )
        train_df, val_df, test_df = splitter.process_splits()
        logger.info("Dataset splitting completed successfully")
        
    except Exception as e:
        logger.error(f"Dataset splitting failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()