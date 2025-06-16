#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# create_ted_sf_mapping.sh
# david miller
#
# Build a TED-ID → CATH-SF mapping from a FASTA file and TSV dictionary.
# ----------------------------------------------------------------------------

set -euo pipefail

# Default paths for input and output files.
# These can be overridden with command-line options.
FASTA_FILE="/SAN/orengolab/cath_alphafold/cath_ted_gold_sequences_hmmvalidated_qscore_0.7_S100_rep_seq.fasta"
DICT_FILE="/state/partition2/NO_BACKUP/databases/ted/datasets/ted_365m.domain_summary.cath.globularity.taxid.qscore.tsv"
OUTPUT_JSON="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-mapping.json"
UNMAPPED_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-unmapped.txt"

usage() {
  cat <<EOF
Usage: ${0##*/} [-f FASTA] [-d TSV_DICT] [-o JSON_OUT] [-u UNMAPPED] [-h]

Build a TED-ID to CATH-SF mapping from FASTA headers and a dictionary file.

This script efficiently maps TED IDs from a FASTA file to CATH Superfamily
identifiers from a large TSV dictionary, producing a JSON output and a list
of unmapped IDs.

Options:
  -f FASTA        Input FASTA file with TED IDs.
                  (default: $FASTA_FILE)
  -d TSV_DICT     Input TSV dictionary with TED-ID in column 1 and CATH-SF
                  in column 15.
                  (default: $DICT_FILE)
  -o JSON_OUT     Output JSON mapping file.
                  (default: $OUTPUT_JSON)
  -u UNMAPPED     Output file for unmapped TED IDs.
                  (default: $UNMAPPED_FILE)
  -h              Print this help message and exit.
EOF
  # Exit with the code provided by the caller (default 0 for normal help).
  local exit_code=${1:-0}
  exit "$exit_code"
}

# Parse command-line options
while getopts ":f:d:o:u:h" opt; do
  case "$opt" in
    f) FASTA_FILE=$OPTARG ;;
    d) DICT_FILE=$OPTARG ;;
    o) OUTPUT_JSON=$OPTARG ;;
    u) UNMAPPED_FILE=$OPTARG ;;
    h) usage 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage 1 ;;
  esac
done

# Validate that input files are readable
for f in "$FASTA_FILE" "$DICT_FILE"; do
  if [[ ! -r "$f" ]]; then
    echo "ERROR: Cannot read input file: $f" >&2
    exit 1
  fi
done

# Ensure output directories exist before writing to them
mkdir -p "$(dirname "$OUTPUT_JSON")" "$(dirname "$UNMAPPED_FILE")" || {
  echo "ERROR: Cannot create output directories" >&2
  exit 1
}

echo "Processing TED-SF mapping..." >&2

# Create a temporary file to hold the sorted, unique TED IDs from the FASTA file.
# Using a temp file is efficient for large lists of IDs.
TEMP_TED_IDS=$(mktemp)
# Set a trap to ensure the temporary file is removed on script exit, even if errors occur.
trap 'rm -f "$TEMP_TED_IDS"' EXIT

# Extract TED IDs from FASTA headers. The ID is the first word after the '>'.
# Sort -u ensures we have a unique list of IDs to look for.
awk '/^>/ { print substr($1, 2) }' "$FASTA_FILE" | sort -u > "$TEMP_TED_IDS"

# Use a single, efficient awk script to perform the mapping and generate all outputs.
# This avoids reading the large dictionary file multiple times.
awk -F '\t' -v ted_ids_file="$TEMP_TED_IDS" \
    -v unmapped_file="$UNMAPPED_FILE" \
    -v output_json_path="$OUTPUT_JSON" \
'
BEGIN {
    # Load required TED IDs from the temp file into a lookup table for fast access.
    while ((getline ted_id < ted_ids_file) > 0) {
        needed_ted_ids[ted_id] = 1
        total_ids++
    }
    close(ted_ids_file)

    # Initialize JSON output and state variables.
    print "{"
    first_entry = 1
    mapped_count = 0

    # Ensure the unmapped IDs file is empty before we start.
    printf "" > unmapped_file
}

# Process the main dictionary file line-by-line.
# A valid line must have at least 15 columns for the CATH SF.
NF >= 15 {
    ted_id = $1
    cath_sf = $15

    # Check if the TED ID is one we need and if it has a valid CATH SF.
    if (ted_id in needed_ted_ids && cath_sf != "" && cath_sf != "-") {
        # Mark ID as found. This handles duplicates and helps find unmapped IDs later.
        delete needed_ted_ids[ted_id]

        # Add a comma before all but the first JSON entry.
        if (!first_entry) {
            printf ",\n"
        }
        first_entry = 0

        # Print the formatted JSON key-value pair.
        printf "  \"%s\": \"%s\"", ted_id, cath_sf
        mapped_count++
    }
}

END {
    # Add a newline after the last JSON entry before closing the object.
    if (mapped_count > 0) {
        printf "\n"
    }
    print "}"

    # Write any remaining (unmapped) TED IDs to the specified file.
    for (ted_id in needed_ted_ids) {
        print ted_id >> unmapped_file
    }
    close(unmapped_file)

    # Calculate final statistics for the run.
    unmapped_count = total_ids - mapped_count
    coverage = (total_ids > 0) ? (mapped_count / total_ids) * 100 : 0

    # Print a summary report to standard error for the user.
    printf "\nSummary\n-------\n" > "/dev/stderr"
    printf "Total TED IDs:    %d\n", total_ids > "/dev/stderr"
    printf "Mapped:           %d\n", mapped_count > "/dev/stderr"
    printf "Unmapped:         %d\n", unmapped_count > "/dev/stderr"
    printf "Coverage:         %.1f%%\n", coverage > "/dev/stderr"
    printf "-----------------\n" > "/dev/stderr"
    printf "Output JSON:      %s\n", output_json_path > "/dev/stderr"
    printf "Unmapped IDs:     %s\n", unmapped_file > "/dev/stderr"
}' "$DICT_FILE" > "$OUTPUT_JSON"

echo "Done." >&2