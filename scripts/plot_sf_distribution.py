import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json


def plot_sf_distribution(json_path, output_dir):
    """
    Generates and saves a histogram of superfamily sizes from a JSON mapping file.

    Args:
        json_path (str): The path to the JSON mapping file.
        output_dir (str): The directory to save the plot in.
    """
    try:
        with open(json_path, 'r') as f:
            sf_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Exception type: {type(e)}")
        print(f"Exception details: {repr(e)}")
        return

    plt.figure(figsize=(12, 6))

    # Calculate FunFam sizes from the SF mapping
    sf_labels = list(sf_mapping.values())
    sf_sizes = pd.Series(sf_labels).value_counts()
    sfs_ge_3 = (sf_sizes >= 3).sum()

    # Create log-scale histogram
    plt.hist(np.log10(sf_sizes), bins=100,
                color='lightblue', alpha=0.8)

    # Remove chart junk
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add statistics
    stats_text = (
        f"Total Superfamilies (n) = {len(sf_sizes):,}\n"
        f"Superfamilies (size ≥ 3) = {sfs_ge_3:,}\n"
        f"Median Size = {sf_sizes.median():.0f}\n"
        f"Mean Size = {sf_sizes.mean():.0f}\n"
        f"Min Size = {sf_sizes.min():.0f}\n"
        f"Max Size = {sf_sizes.max():.0f}"
    )

    plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('log10(Sequences per TED s100 SF)', fontsize=12)
    plt.ylabel('Number of TED s100 SFs', fontsize=12)
    plt.title('Distribution of TED s100 SF Sizes', fontsize=14)

    plt.tight_layout()

    # Save the figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "TED_s100_sf_size_distribution.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    plt.show()


def main():
    """
    Main function to parse arguments and call the plotting function.
    """
    parser = argparse.ArgumentParser(description="Plot superfamily size distribution from a JSON mapping file.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="data/TED/TED-SF-mapping.json",
        help="Path to the JSON file containing sequence to SF mapping."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the output plot."
    )
    args = parser.parse_args()

    plot_sf_distribution(args.json_path, args.output_dir)


if __name__ == "__main__":
    main() 