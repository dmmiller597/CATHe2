#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# create_ted_sf_mapping.sh
# david miller
#
# Build a TED-ID → CATH-SF mapping from a FASTA file and TSV dictionary.
# ----------------------------------------------------------------------------

set -euo pipefail

# Default paths
FASTA_FILE="/SAN/orengolab/cath_alphafold/cath_ted_gold_sequences_hmmvalidated_qscore_0.7_S100_rep_seq.fasta"
DICT_FILE="/state/partition2/NO_BACKUP/databases/ted/datasets/ted_365m.domain_summary.cath.globularity.taxid.qscore.tsv"
OUTPUT_JSON="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-mapping.json"
UNMAPPED_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-unmapped.txt"

usage() {
  cat <<EOF
Usage: ${0##*/} [-f FASTA] [-d TSV_DICT] [-o JSON_OUT] [-u UNMAPPED] [-h]

Build a TED-ID to CATH-SF mapping from FASTA headers and dictionary.

Options:
  -f FASTA        Input FASTA with TED IDs (default: $FASTA_FILE)
  -d TSV_DICT     TSV with TED-ID (col-1) and CATH-SF (col-15) (default: $DICT_FILE)
  -o JSON_OUT     Output JSON mapping file (default: $OUTPUT_JSON)
  -u UNMAPPED     Output unmapped IDs file (default: $UNMAPPED_FILE)
  -h              Print this help and exit
EOF
  exit 1
}

# Parse CLI options
while getopts ":f:d:o:u:h" opt; do
  case "$opt" in
    f) FASTA_FILE=$OPTARG ;;
    d) DICT_FILE=$OPTARG ;;
    o) OUTPUT_JSON=$OPTARG ;;
    u) UNMAPPED_FILE=$OPTARG ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
  esac
done

# Validate inputs
for f in "$FASTA_FILE" "$DICT_FILE"; do
  if [[ ! -r "$f" ]]; then
    echo "ERROR: Cannot read input file: $f" >&2
    exit 1
  fi
done

# Ensure output directories exist
mkdir -p "$(dirname "$OUTPUT_JSON")" "$(dirname "$UNMAPPED_FILE")" || {
  echo "ERROR: Cannot create output directories" >&2
  exit 1
}

echo "Processing TED-SF mapping..." >&2

# Extract TED IDs from FASTA and create sorted temp file for efficient lookup
TEMP_TED_IDS=$(mktemp)
trap 'rm -f "$TEMP_TED_IDS"' EXIT

awk '/^>/ {
  # Extract TED ID (everything after > up to first space)
  ted_id = substr($1, 2)
  print ted_id
}' "$FASTA_FILE" | sort -u > "$TEMP_TED_IDS"

# Process TSV and generate JSON directly using awk for efficiency
awk -v ted_ids_file="$TEMP_TED_IDS" -v unmapped_file="$UNMAPPED_FILE" '
BEGIN {
  # Load TED IDs into lookup table
  while ((getline ted_id < ted_ids_file) > 0) {
    needed_ted_ids[ted_id] = 1
    total_ids++
  }
  close(ted_ids_file)
  
  # Initialize output
  print "{"
  first = 1
  mapped_count = 0
  
  # Clear unmapped file
  print "" > unmapped_file
  close(unmapped_file)
}

# Process TSV file
FNR > 0 && NF >= 15 {
  ted_id = $1
  cath_sf = $15
  
  # Only process if this TED ID is needed and has valid CATH SF
  if (ted_id in needed_ted_ids && cath_sf != "" && cath_sf != "-") {
    # Mark as found and output JSON entry
    delete needed_ted_ids[ted_id]
    if (!first) print ","
    first = 0
    
    # Escape quotes in values
    gsub(/"/, "\\\"", ted_id)
    gsub(/"/, "\\\"", cath_sf)
    
    printf "  \"%s\": \"%s\"", ted_id, cath_sf
    mapped_count++
  }
}

END {
  # Close JSON
  print ""
  print "}"
  
  # Write unmapped TED IDs
  for (ted_id in needed_ted_ids) {
    print ted_id >> unmapped_file
  }
  close(unmapped_file)
  
  # Calculate statistics
  unmapped_count = total_ids - mapped_count
  coverage = (total_ids > 0) ? (mapped_count / total_ids) * 100 : 0
  
  # Print summary to stderr
  printf "\nSummary\n-------\n" > "/dev/stderr"
  printf "Total TED IDs:    %d\n", total_ids > "/dev/stderr"
  printf "Mapped:           %d\n", mapped_count > "/dev/stderr"
  printf "Unmapped:         %d\n", unmapped_count > "/dev/stderr"
  printf "Coverage:         %.1f%%\n", coverage > "/dev/stderr"
  printf "-----------------\n" > "/dev/stderr"
  printf "Output JSON:      %s\n", ENVIRON["OUTPUT_JSON"] > "/dev/stderr"
  printf "Unmapped IDs:     %s\n", ENVIRON["UNMAPPED_FILE"] > "/dev/stderr"
}' OUTPUT_JSON="$OUTPUT_JSON" UNMAPPED_FILE="$UNMAPPED_FILE" "$DICT_FILE" > "$OUTPUT_JSON"

echo "Done." >&2