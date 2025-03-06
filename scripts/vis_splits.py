from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
import argparse
matplotlib.use('Agg')  # Prevent display issues on HPC

def setup_subplot_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str = '') -> None:
    """Configure common styling for a subplot."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
def plot_length_distribution(ax: plt.Axes, data: pd.DataFrame, name: str, show_ylabel: bool = False) -> None:
    """Plot sequence length distribution for a given dataset."""
    ax.hist(data['length'], bins=80, color='lightblue', alpha=0.8)
    setup_subplot_style(
        ax, 
        f'{name} Length Distribution',
        'Sequence Length (amino acids)',
        'Number of Sequences' if show_ylabel else ''
    )
    
    # Add statistics
    stats_text = (
        f"n = {len(data):,}\n"
        f"median = {data['length'].median():.0f}\n"
        f"mean = {data['length'].mean():.0f}\n"
        f"min = {data['length'].min():.0f}\n"
        f"max = {data['length'].max():.0f}"
    )
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            va='top', ha='right',
            bbox=dict(facecolor='white', alpha=0.8))

def plot_sf_size_distribution(ax: plt.Axes, data: pd.DataFrame, name: str, show_ylabel: bool = False) -> None:
    """Plot SF size distribution for a given dataset."""
    sf_sizes = data.groupby('SF').size()
    ax.hist(np.log10(sf_sizes), bins=50, color='lightblue', alpha=0.8)
    setup_subplot_style(
        ax,
        f'{name} SF Size Distribution',
        'log10(Sequences per SF)',
        'Number of SFs' if show_ylabel else ''
    )
    
    # Add statistics
    stats_text = (
        f"n SFs = {len(sf_sizes):,}\n"
        f"median size = {sf_sizes.median():.0f}\n"
        f"mean size = {sf_sizes.mean():.0f}\n"
        f"min size = {sf_sizes.min():.0f}\n"
        f"max size = {sf_sizes.max():.0f}"
    )
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            va='top', ha='right',
            bbox=dict(facecolor='white', alpha=0.8))

def plot_dataset_distributions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Create a comprehensive visualization of sequence length and SF size distributions.
    
    Args:
        train_df: Training dataset
        val_df: Validation dataset
        test_df: Test dataset
        output_path: Path to save the plot
    """
    # Create figure with 2 rows of subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot distributions
    datasets = [('Train', train_df), ('Validation', val_df), ('Test', test_df)]
    
    # Row 1: Length distributions
    for i, (name, data) in enumerate(datasets):
        plot_length_distribution(
            [ax1, ax2, ax3][i],
            data,
            name,
            show_ylabel=(i == 0)
        )
    
    # Row 2: SF size distributions
    for i, (name, data) in enumerate(datasets):
        plot_sf_size_distribution(
            [ax4, ax5, ax6][i],
            data,
            name,
            show_ylabel=(i == 0)
        )
    
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for name, data in datasets:
        n_seqs = len(data)
        n_sfs = data['SF'].nunique()
        print(f"\n{name}:")
        print(f"Number of sequences: {n_seqs:,}")
        print(f"Number of unique SFs: {n_sfs:,}")
        print(f"Average sequences per SF: {n_seqs/n_sfs:.1f}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize sequence and SF distributions across dataset splits.')
    parser.add_argument('--train', type=str, default="data/TED/s30/s30_train.parquet",
                        help='Path to training data parquet file')
    parser.add_argument('--val', type=str, default="data/TED/s30/s30_val.parquet",
                        help='Path to validation data parquet file')
    parser.add_argument('--test', type=str, default="data/TED/s30/s30_test.parquet",
                        help='Path to test data parquet file')
    parser.add_argument('--output', type=str, default="figures/dataset_distributions.png",
                        help='Path to save the output figure')
    
    args = parser.parse_args()
    
    # Load data from parquet files
    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    test_df = pd.read_parquet(args.test)
    
    # Add sequence length column to each DataFrame
    for df in [train_df, val_df, test_df]:
        # Replace 'sequence' with your actual sequence column name
        df['length'] = df['sequence'].str.len()
    
    plot_dataset_distributions(
        train_df,
        val_df,
        test_df,
        output_path=Path(args.output)
    )