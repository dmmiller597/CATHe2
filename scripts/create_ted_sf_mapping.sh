#!/usr/bin/env bash
# david miller - 16/06/2025
# Create TED-ID to CATH-SF mapping from FASTA and TSV files
# Efficiently handles millions of sequences using awk.
# need to run on bubba213-3 
set -euo pipefail
IFS=$'\n\t'

# Input files
FASTA_FILE="/SAN/orengolab/cath_alphafold/cath_ted_gold_sequences_hmmvalidated_qscore_0.7_S100_rep_seq.fasta"
DICT_FILE="/state/partition2/NO_BACKUP/databases/ted/datasets/ted_365m.domain_summary.cath.globularity.taxid.qscore.tsv"

# Output files
OUTPUT_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-mapping.json"
UNMAPPED_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-unmapped.txt"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$UNMAPPED_FILE")"

# Temporary files
TED_IDS_FILE=$(mktemp -t ted_ids.XXXXXX)
TSV_LOOKUP_FILE=$(mktemp -t ted_lookup.XXXXXX)

# Cleanup function
cleanup() {
    rm -f "$TED_IDS_FILE" "$TSV_LOOKUP_FILE" "${TEMP_FASTA_FOR_TEST:-}" "${TEMP_DICT_FOR_TEST:-}"
}
trap cleanup EXIT

echo "Processing files..."

# Extract TED-IDs from FASTA headers
echo "Extracting TED-IDs from FASTA file..."
awk '/^>/{sub(/^>/,""); sub(/\r$/,"\"); print}' "$FASTA_FILE" > "$TED_IDS_FILE"
TOTAL_FASTA_IDS=$(wc -l < "$TED_IDS_FILE")
echo "Found $TOTAL_FASTA_IDS TED-IDs in FASTA file"

# Create lookup table from TSV (TED-ID -> CATH-SF)
echo "Creating lookup table from TSV file..."
sed 's/\r$//' "$DICT_FILE" | awk -F '\t' 'NF >= 15 && $15 != "-" && $15 != "" { print $1 "\t" $15 }' > "$TSV_LOOKUP_FILE"
TOTAL_TSV_MAPPINGS=$(wc -l < "$TSV_LOOKUP_FILE")
echo "Found $TOTAL_TSV_MAPPINGS TED-IDs with CATH-SF mappings in TSV file"

# Use awk to perform the mapping efficiently.
# Mapped JSON is printed to stdout, unmapped IDs to stderr.
echo "Creating JSON mapping using awk..."
{
    awk -F'\t' '
        BEGIN { print "{" }
        NR==FNR {
            lookup[$1] = $2
            next
        }
        {
            if ($1 in lookup) {
                if (!first_entry) printf ",\n"
                first_entry = 0
                val = lookup[$1]
                gsub(/"/, "\\\"", val)
                printf "  \"%s\": \"%s\"", $1, val
            } else {
                print $1 > "/dev/stderr"
            }
        }
        END {
            if (first_entry == 0) printf "\n"
            print "}"
        }
    ' "$TSV_LOOKUP_FILE" "$TED_IDS_FILE"
} > "$OUTPUT_FILE" 2> "$UNMAPPED_FILE"

echo "Mapping complete."

# Calculate counts from the generated files for the summary
MAPPED_COUNT=$(grep -c '":' "$OUTPUT_FILE")
UNMAPPED_COUNT=$(wc -l < "$UNMAPPED_FILE" 2>/dev/null || echo 0)

# Display summary
echo ""
echo "=== SUMMARY ==="
echo "Total FASTA TED-IDs: $TOTAL_FASTA_IDS"
echo "Successfully mapped: $MAPPED_COUNT"
echo "Unmapped: $UNMAPPED_COUNT"
if [[ $TOTAL_FASTA_IDS -gt 0 ]]; then
    # Use awk for floating point arithmetic for a more precise percentage
    PERCENTAGE=$(awk -v mapped="$MAPPED_COUNT" -v total="$TOTAL_FASTA_IDS" 'BEGIN { printf "%.2f", (mapped / total) * 100 }')
    echo "Mapping coverage: ${PERCENTAGE}%"
fi
echo ""
echo "JSON mapping written to: $OUTPUT_FILE"
echo "Unmapped IDs written to: $UNMAPPED_FILE"
echo "Script completed successfully!"
