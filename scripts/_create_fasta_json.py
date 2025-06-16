import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from tqdm import tqdm


def create_fasta_and_json(input_file: Path, output_dir: Path):
    """
    Reads a parquet file containing protein sequence data and creates a FASTA
    file and a corresponding JSON metadata file.

    This function processes the parquet file in chunks to be memory-efficient.

    Args:
        input_file: Path to the input parquet file.
        output_dir: Path to the directory where output files will be saved.
    """
    if not input_file.is_file():
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_file.stem
    fasta_file_path = output_dir / f"{base_name}.fasta"
    json_file_path = output_dir / f"{base_name}.json"

    metadata = {}

    try:
        parquet_file = pq.ParquetFile(input_file)
        num_rows = parquet_file.metadata.num_rows

        print(f"Processing {num_rows} rows from {input_file}...")

        with open(fasta_file_path, "w") as fasta_file:
            with tqdm(
                total=num_rows, desc="Creating FASTA and JSON", unit="seq"
            ) as pbar:
                for batch in parquet_file.iter_batches(batch_size=20000):
                    df = batch.to_pandas()
                    for row in df.itertuples():
                        seq_id = row.sequence_id
                        sequence = row.sequence
                        sf_val = row.SF
                        split_val = row.split

                        fasta_file.write(f">{seq_id}\n")
                        fasta_file.write(f"{sequence}\n")

                        metadata[seq_id] = {"SF": sf_val, "split": split_val}
                    pbar.update(len(df))

        print(f"Writing metadata to JSON file: {json_file_path}")
        with open(json_file_path, "w") as json_file:
            json.dump(metadata, json_file, indent=2)

        print("\nProcessing complete.")
        print(f"FASTA file saved to: {fasta_file_path}")
        print(f"JSON file saved to: {json_file_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Create FASTA and JSON files from a parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        default="data/TED/s30/s30_full.parquet",
        help="Path to the input parquet file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="data/TED/s30",
        help="Path to the output directory.",
    )

    args = parser.parse_args()

    create_fasta_and_json(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
