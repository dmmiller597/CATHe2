import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch import Tensor
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

log = logging.getLogger(__name__)

def generate_tsne_plot(self, embeddings: Tensor, labels: Tensor) -> None:
    """
    Creates a minimalist visualization of embeddings using PCA preprocessing
    followed by t-SNE, colored by CATH classes, following Tufte principles.

    Args:
        embeddings: Projected embeddings tensor
        labels: Corresponding label tensor (CATH class IDs)
    """
    try:
        # Start timing
        start_time = time.time()
        log.info(f"Generating PCA+t-SNE plot for epoch {self.current_epoch}")

        # Use a local Generator for reproducible sampling without altering global RNG
        gen = torch.Generator(device=embeddings.device).manual_seed(self.hparams.seed)

        # Sample for efficiency (max 10000 points) with local generator
        max_samples = 10000
        indices = torch.randperm(len(embeddings), generator=gen)[:max_samples]
        emb_subset, lbl_subset = embeddings[indices], labels[indices]

        # Get unique labels
        unique_labels = np.unique(lbl_subset)

        cath_class_names = {
            0: "Mainly Alpha",
            1: "Mainly Beta",
            2: "Alpha Beta",
            3: "Few Secondary Structures"
        }

        # --- PCA Preprocessing ---
        n_components_pca = min(50, emb_subset.shape[1]) # Limit PCA components
        log.debug(f"Running PCA with n_components={n_components_pca}")
        pca = PCA(n_components=n_components_pca, random_state=42)
        with warnings.catch_warnings(): # Suppress potential future warnings
                warnings.simplefilter("ignore", category=FutureWarning)
                embeddings_pca = pca.fit_transform(emb_subset)
        log.debug("PCA completed.")

        # --- t-SNE on PCA results ---
        log.debug("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=1) # n_jobs=1 for reproducibility with random_state
        with warnings.catch_warnings(): # Suppress potential future warnings
                warnings.simplefilter("ignore", category=UserWarning) # e.g., for n_jobs override
                warnings.simplefilter("ignore", category=FutureWarning)
                tsne_result = tsne.fit_transform(embeddings_pca)
        log.debug("t-SNE completed.")

        # Set up minimalist plot with Tufte-inspired style
        plt.figure(figsize=(8, 8))

        # Set clean style
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = False

        # Create color mapping that maintains consistent colors per CATH class
        color_palette = sns.color_palette("colorblind", n_colors=len(unique_labels))
        # Handle cases where labels might not start from 0 or be consecutive
        label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
        colors = [color_palette[label_to_color_idx[int(label)]] for label in lbl_subset]


        # Create scatter plot - smaller points with higher density
        plt.scatter(
            tsne_result[:, 0], tsne_result[:, 1],
            c=colors,
            s=5,  # Smaller point size
            alpha=0.7,  # Slightly transparent
            linewidths=0,  # No edge lines
            rasterized=True  # Better for export
        )

        # Minimal labels and subtle tick marks
        plt.title(f'CATH Classes (Epoch {self.current_epoch}) - PCA+tSNE', fontsize=12, pad=10) # Adjusted title
        plt.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.5)

        # Subtle axis labels
        plt.xlabel('t-SNE Dimension 1 (via PCA)', fontsize=9, labelpad=7, color='#505050')
        plt.ylabel('t-SNE Dimension 2 (via PCA)', fontsize=9, labelpad=7, color='#505050')

        # Create minimal legend with CATH class names
        if len(unique_labels) <= 4:
            handles = [plt.Line2D([0], [0], marker='o', color=color_palette[label_to_color_idx[label]],
                                    markersize=5, linestyle='',
                                    label=cath_class_names.get(label, f"Class {label}"))
                        for label in unique_labels]
            plt.legend(handles=handles,
                        loc='best',
                        frameon=False,  # No frame
                        fontsize=9,
                        handletextpad=0.5)  # Less space between marker and text

        # Tighter layout with reduced margins
        plt.tight_layout(pad=1.2)

        # Save the plot
        filename = f"tsne_pca_epoch_{self.current_epoch}.png"
        save_path = os.path.join(self.hparams.tsne_viz_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

        # Log completion
        elapsed_time = time.time() - start_time
        log.info(f"PCA+t-SNE plot saved to {save_path} (took {elapsed_time:.2f}s)")

    except Exception as e:
        log.error(f"Error generating PCA+t-SNE plot: {e}")
        log.exception("Detailed traceback:")

def generate_umap_plot(self, embeddings: Tensor, labels: Tensor) -> None:
    """
    Creates a minimalist visualization of embeddings using UMAP,
    colored by CATH classes, following Tufte principles.

    Args:
        embeddings: Projected embeddings tensor
        labels: Corresponding label tensor (CATH class IDs)
    """
    try:
        # Start timing
        start_time = time.time()
        log.info(f"Generating UMAP plot for epoch {self.current_epoch}")

        # Use a local Generator for reproducible sampling without altering global RNG
        gen = torch.Generator(device=embeddings.device).manual_seed(self.hparams.seed)

        # Sample for efficiency (max 10000 points) with local generator
        max_samples = 10000
        indices = torch.randperm(len(embeddings), generator=gen)[:max_samples]
        emb_subset, lbl_subset = embeddings[indices], labels[indices]

        # Get unique labels
        unique_labels = np.unique(lbl_subset)

        cath_class_names = {
            0: "Mainly Alpha",
            1: "Mainly Beta",
            2: "Alpha Beta",
            3: "Few Secondary Structures"
        }

        # --- UMAP ---
        reducer = umap.UMAP(
            n_neighbors=15, # Default, balances local/global structure
            min_dist=0.1,  # Default, controls tightness of clusters
            n_components=2,
            metric='euclidean', # Use Euclidean distance
            random_state=42,
            # Consider low_memory=True for very large datasets if memory is an issue
        )
        umap_result = reducer.fit_transform(emb_subset)

        # Set up minimalist plot with Tufte-inspired style
        plt.figure(figsize=(8, 8))

        # Set clean style
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = False

        # Create color mapping that maintains consistent colors per CATH class
        color_palette = sns.color_palette("colorblind", n_colors=len(unique_labels))
        # Handle cases where labels might not start from 0 or be consecutive
        label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
        colors = [color_palette[label_to_color_idx[int(label)]] for label in lbl_subset]

        # Create scatter plot - smaller points with higher density
        plt.scatter(
            umap_result[:, 0], umap_result[:, 1],
            c=colors,
            s=5,  # Smaller point size
            alpha=0.7,  # Slightly transparent
            linewidths=0,  # No edge lines
            rasterized=True  # Better for export
        )

        # Minimal labels and subtle tick marks
        plt.title(f'CATH Classes (Epoch {self.current_epoch}) - UMAP', fontsize=12, pad=10)
        plt.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.5)

        # Subtle axis labels
        plt.xlabel('UMAP Dimension 1', fontsize=9, labelpad=7, color='#505050')
        plt.ylabel('UMAP Dimension 2', fontsize=9, labelpad=7, color='#505050')

        # Create minimal legend with CATH class names
        if len(unique_labels) <= 4: # Only show legend if few classes
            handles = [plt.Line2D([0], [0], marker='o', color=color_palette[label_to_color_idx[label]],
                                    markersize=5, linestyle='',
                                    label=cath_class_names.get(label, f"Class {label}"))
                        for label in unique_labels]
            plt.legend(handles=handles,
                        loc='best',
                        frameon=False,  # No frame
                        fontsize=9,
                        handletextpad=0.5)  # Less space between marker and text

        # Tighter layout with reduced margins
        plt.tight_layout(pad=1.2)

        # Save the plot
        filename = f"umap_epoch_{self.current_epoch}.png"
        save_path = os.path.join(self.hparams.umap_viz_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

        # Log completion
        elapsed_time = time.time() - start_time
        log.info(f"UMAP plot saved to {save_path} (took {elapsed_time:.2f}s)")

    except Exception as e:
        log.error(f"Error generating UMAP plot: {e}")
        log.exception("Detailed traceback:")