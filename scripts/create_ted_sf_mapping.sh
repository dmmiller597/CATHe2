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

# Start timing
START_TIME=$(date +%s)

echo "=== TED-ID to CATH-SF Mapping Tool ===" >&2
echo "Started at: $(date)" >&2
echo >&2

# Validate inputs and show file information
echo "Validating input files..." >&2
for f in "$FASTA_FILE" "$DICT_FILE"; do
  if [[ ! -r "$f" ]]; then
    echo "ERROR: Cannot read input file: $f" >&2
    exit 1
  fi
done

echo "Input files:" >&2
echo "  FASTA: $(basename "$FASTA_FILE")" >&2
echo "    Size: $(du -h "$FASTA_FILE" | cut -f1)" >&2
echo "    Sequences: $(grep -c '^>' "$FASTA_FILE" 2>/dev/null || echo "unknown")" >&2
echo "  Dictionary: $(basename "$DICT_FILE")" >&2
echo "    Size: $(du -h "$DICT_FILE" | cut -f1)" >&2
echo "    Records: $(wc -l < "$DICT_FILE" 2>/dev/null || echo "unknown")" >&2
echo >&2

# Ensure output directories exist
echo "Creating output directories..." >&2
mkdir -p "$(dirname "$OUTPUT_JSON")" "$(dirname "$UNMAPPED_FILE")" || {
  echo "ERROR: Cannot create output directories" >&2
  exit 1
}

echo "Output files:" >&2
echo "  JSON mapping: $OUTPUT_JSON" >&2
echo "  Unmapped IDs: $UNMAPPED_FILE" >&2
echo >&2

echo "Processing pipeline:" >&2
echo "  [1/4] Extracting TED IDs from FASTA headers..." >&2
echo "  [2/4] Extracting CATH-SF mappings from dictionary..." >&2
echo "  [3/4] Joining data and generating JSON..." >&2
echo "  [4/4] Writing output files..." >&2
echo >&2

# Extract TED IDs from FASTA and join with dictionary to create JSON
{
  echo "{"
  join -t$'\t' -a 1 \
    <(awk '/^>/ {sub(/^>/,""); print $1}' "$FASTA_FILE" | LC_ALL=C sort -u) \
    <(awk -F'\t' '$15 != "" && $15 != "-" {print $1 "\t" $15}' "$DICT_FILE" | LC_ALL=C sort -k1,1) |
  awk -v unmapped="$UNMAPPED_FILE" '
    BEGIN { 
      first = 1
      mapped_count = 0
      unmapped_count = 0
    }
    {
      if (NF == 2) {
        if (!first) printf ",\n"
        first = 0
        gsub(/"/, "\\\"", $2)  # Escape quotes in SF value
        printf "  \"%s\": \"%s\"", $1, $2
        mapped_count++
        if (mapped_count % 1000 == 0) {
          printf "\rProcessing... %d mapped so far", mapped_count > "/dev/stderr"
        }
      } else {
        print $1 > unmapped
        unmapped_count++
      }
    }
    END {
      if (mapped_count > 0) {
        printf "\rProcessing... %d total mapped, %d unmapped\n", mapped_count, unmapped_count > "/dev/stderr"
      }
    }'
  echo -e "\n}"
} > "$OUTPUT_JSON"

echo "Finalizing output files..." >&2

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

# Calculate and display statistics
TOTAL_IDS=$(awk '/^>/ {count++} END {print count+0}' "$FASTA_FILE")
MAPPED_COUNT=$(awk -F'"' 'NF > 2 {count++} END {print count+0}' "$OUTPUT_JSON")
UNMAPPED_COUNT=$((TOTAL_IDS - MAPPED_COUNT))
COVERAGE=$(awk -v m="$MAPPED_COUNT" -v t="$TOTAL_IDS" 'BEGIN {
  if (t > 0) printf "%.1f", (m/t)*100; else print "0.0"
}')

cat >&2 <<EOF

=== RESULTS ===
Total TED IDs:    $TOTAL_IDS
Mapped:           $MAPPED_COUNT (${COVERAGE}%)
Unmapped:         $UNMAPPED_COUNT
-----------------
Output JSON:      $OUTPUT_JSON ($(du -h "$OUTPUT_JSON" | cut -f1))
Unmapped IDs:     $UNMAPPED_FILE ($(wc -l < "$UNMAPPED_FILE" 2>/dev/null || echo 0) lines)
-----------------
Processing time:  ${ELAPSED_MIN}m ${ELAPSED_SEC}s
Completed at:     $(date)
EOF

echo >&2
echo "Done! ✓" >&2