#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# create_ted_sf_mapping.sh
# david miller
#
# Build a TED-ID → CATH-SF mapping from a FASTA file and TSV dictionary.
# Processes data in a streaming fashion to minimize memory usage.
# ----------------------------------------------------------------------------

set -euo pipefail

# Default paths
FASTA_FILE="/SAN/orengolab/cath_alphafold/cath_ted_gold_sequences_hmmvalidated_qscore_0.7_S100_rep_seq.fasta"
DICT_FILE="/state/partition2/NO_BACKUP/databases/ted/datasets/ted_365m.domain_summary.cath.globularity.taxid.qscore.tsv"
OUTPUT_JSON="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-mapping.json"
UNMAPPED_IDS="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-unmapped.txt"
TMP_PARENT="${TMPDIR:-/tmp}"

usage() {
  cat <<EOF
Usage: ${0##*/} [-f FASTA] [-d TSV_DICT] [-o JSON_OUT] [-u UNMAPPED] [-t TMP_PARENT] [-h]

Build a TED-ID to CATH-SF mapping from FASTA headers and dictionary.

Options:
  -f FASTA        Input FASTA with TED IDs (default: $FASTA_FILE)
  -d TSV_DICT     TSV with TED-ID (col-1) and CATH-SF (col-15) (default: $DICT_FILE)
  -o JSON_OUT     Output JSON mapping file (default: $OUTPUT_JSON)
  -u UNMAPPED     Output unmapped IDs file (default: $UNMAPPED_IDS)
  -t TMP_PARENT   Directory for temporary files (default: env TMPDIR or /tmp)
  -h              Print this help and exit
EOF
  exit 1
}

# Parse CLI options
while getopts ":f:d:o:u:t:h" opt; do
  case "$opt" in
    f) FASTA_FILE=$OPTARG ;;
    d) DICT_FILE=$OPTARG ;;
    o) OUTPUT_JSON=$OPTARG ;;
    u) UNMAPPED_IDS=$OPTARG ;;
    t) TMP_PARENT=$OPTARG ;;
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
for d in "$(dirname "$OUTPUT_JSON")" "$(dirname "$UNMAPPED_IDS")"; do
  mkdir -p "$d" || { echo "ERROR: Cannot create output directory $d" >&2; exit 1; }
done

if [[ ! -d "$TMP_PARENT" || ! -w "$TMP_PARENT" ]]; then
  echo "ERROR: TMP dir is not a writable directory: $TMP_PARENT" >&2
  exit 1
fi

# Setup temporary workspace
SCRATCH="$(mktemp -d "$TMP_PARENT/tedmap.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT INT TERM

# Process in a single pipeline
echo "Processing FASTA headers and creating mapping..." >&2

# Extract TED IDs from FASTA and join with dictionary to create JSON
{
  echo "{"
  join -t$'\t' -a 1 \
    <(awk '/^>/ {sub(/^>/,""); print $1}' "$FASTA_FILE" | LC_ALL=C sort -u) \
    <(awk -F'\t' '$15 != "" && $15 != "-" {print $1 "\t" $15}' "$DICT_FILE" | LC_ALL=C sort -k1,1) |
  awk -v unmapped="$UNMAPPED_IDS" '
    BEGIN { first = 1 }
    {
      if (NF == 2) {
        if (!first) printf ",\n"
        first = 0
        gsub(/"/, "\\\"", $2)  # Escape quotes in SF value
        printf "  \"%s\": \"%s\"", $1, $2
      } else {
        print $1 > unmapped
      }
    }'
  echo -e "\n}"
} > "$OUTPUT_JSON"

# Calculate and display statistics
TOTAL_IDS=$(grep -c '^>' "$FASTA_FILE")
MAPPED_IDS=$(grep -c ':' "$OUTPUT_JSON")
UNMAPPED_IDS=$((TOTAL_IDS - MAPPED_IDS))
COVERAGE=$(awk -v m="$MAPPED_IDS" -v t="$TOTAL_IDS" 'BEGIN {printf "%.1f", (m/t)*100}')

cat >&2 <<EOF

Summary
-------
Total TED IDs:    $TOTAL_IDS
Mapped:           $MAPPED_IDS
Unmapped:         $UNMAPPED_IDS
Coverage:         ${COVERAGE}%
-----------------
Output JSON:      $OUTPUT_JSON
Unmapped IDs:     $UNMAPPED_IDS
EOF

echo "Done." >&2