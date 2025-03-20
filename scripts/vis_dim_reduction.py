#!/usr/bin/env python3
#  t-SNE embeddings visualization for CATH hierarchy levels

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import seaborn as sns
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set up basic plot aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'

# Set up command line arguments
parser = argparse.ArgumentParser(description='Generate t-SNE visualizations for protein embeddings')
parser.add_argument('--embeddings', type=str, default='data/embeddings/SF_Train_ProtT5.npz',
                    help='Path to the embeddings file (.npz format)')
parser.add_argument('--labels', type=str, default='data/annotations/Y_Train_SF.csv',
                    help='Path to the labels file (.csv format)')
parser.add_argument('--output', type=str, default='results/Train_ProtT5',
                    help='Directory to save visualization results')
args = parser.parse_args()

# File paths from command line arguments
embeddings_file = args.embeddings
labels_file = args.labels
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

# Load embeddings
logger.info("Loading embeddings...")
embeddings_data = np.load(embeddings_file)
embeddings = embeddings_data['embeddings'] if 'embeddings' in embeddings_data else embeddings_data['arr_0']
logger.info(f"Loaded embeddings with shape: {embeddings.shape}")

# Load labels and parse CATH hierarchy
logger.info("Loading labels and parsing CATH hierarchy levels...")
cath_levels = []  # Will contain lists of labels for each level
cath_level_names = ['Class', 'Architecture', 'Topology', 'Homology']
for _ in range(4):  # Initialize 4 empty lists for the 4 CATH levels
    cath_levels.append([])

with open(labels_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # Skip header if present
    sf_idx = header.index('SF')
    
    for row in reader:
        if row:  # Skip empty rows
            cath_code = row[sf_idx]
            parts = cath_code.split('.')
            
            # Store each level of the hierarchy (C, A, T, H)
            for i in range(4):
                if i < len(parts):
                    # For each level, use the concatenated string up to that level
                    level_code = '.'.join(parts[:i+1])
                    cath_levels[i].append(level_code)
                else:
                    # Handle incomplete CATH codes
                    cath_levels[i].append('unknown')

# Check if dimensions match
if len(cath_levels[0]) != embeddings.shape[0]:
    logger.warning(f"Warning: Number of labels ({len(cath_levels[0])}) doesn't match number of embeddings ({embeddings.shape[0]})")
    min_length = min(len(cath_levels[0]), embeddings.shape[0])
    for i in range(4):
        cath_levels[i] = cath_levels[i][:min_length]
    embeddings = embeddings[:min_length]
    logger.info(f"Trimmed to {min_length} samples")

# Apply PCA before t-SNE to reduce computation
logger.info("Applying PCA to reduce dimensions to 50...")
pca = PCA(n_components=50, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)
logger.info(f"Reduced embeddings shape after PCA: {embeddings_pca.shape}")
logger.info(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")

# Run t-SNE on PCA-reduced data
logger.info("Running t-SNE on PCA-reduced data...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)
tsne_result = tsne.fit_transform(embeddings_pca)

# Function to create and save t-SNE plot for a given CATH level
def create_tsne_plot(tsne_result, labels, level_name, output_dir):
    unique_classes = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_classes)}
    numeric_labels = np.array([label_to_id[label] for label in labels])
    
    n_classes = len(unique_classes)
    logger.info(f"Creating t-SNE visualization for {level_name} level with {n_classes} unique groups...")
    
    # Create figure with appropriate aspect ratio
    plt.figure(figsize=(8, 8))
    
    # Setup plot area with minimal non-data ink
    ax = plt.gca()
    ax.grid(False)
    ax.tick_params(axis='both', which='both', length=3, width=0.8, pad=4)
    
    # Use colorblind palette for better accessibility
    color_palette = sns.color_palette("colorblind", n_colors=n_classes)
    colors = [color_palette[i] for i in numeric_labels]
    scatter = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1],
        c=colors,
        s=10,
        alpha=0.8,
        linewidth=0
    )
    
    # More understated title and axis labels
    plt.title(f'CATH {level_name} Level', fontsize=14, pad=10)
    plt.xlabel('t-SNE 1', fontsize=10, labelpad=8)
    plt.ylabel('t-SNE 2', fontsize=10, labelpad=8)
    
    # Add legend if not too many classes, using a more subtle placement
    if n_classes <= 4:
        # Create simple legend with class labels
        handles = [plt.Line2D([0], [0], marker='o', color=color_palette[i], 
                             markersize=6, label=unique_classes[i])
                  for i in range(n_classes)]
        plt.legend(handles=handles,
                  title=f"CATH {level_name}",
                  loc='best',
                  frameon=True,
                  fontsize=10)
    # Adjust axis limits to provide small margin around data points
    x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
    y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()
    margin = 0.05
    plt.xlim(x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
    plt.ylim(y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))
    
    plt.tight_layout()
    filename = f'tsne_embeddings_CATH_{level_name.lower()}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"t-SNE visualization saved to {output_dir}/{filename}")
    plt.close()

# Create plots for each CATH level
for i, level_labels in enumerate(cath_levels):
    create_tsne_plot(tsne_result, level_labels, cath_level_names[i], output_dir)

logger.info("All visualizations completed!")