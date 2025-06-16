#!/bin/bash
# david miller - 16/06/2025
# Create TED-ID to CATH-SF mapping from FASTA and TSV files
# Efficiently handles millions of sequences using awk.
# need to run on bubba213-3 
set -e
set -o pipefail

# Input files
FASTA_FILE="/SAN/orengolab/cath_alphafold/cath_ted_gold_sequences_hmmvalidated_qscore_0.7_S100_rep_seq.fasta"
DICT_FILE="/state/partition2/NO_BACKUP/databases/ted/datasets/ted_365m.domain_summary.cath.globularity.taxid.qscore.tsv"

# Output files
OUTPUT_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-mapping.json"
UNMAPPED_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-unmapped.txt"

# --- START OF TEST BLOCK ---
# This block creates smaller, temporary input files for a quick test run.
# It uses the first 1000 lines from your original files.
# To run the script on the full data, simply delete this entire block.
echo "--- Running in TEST mode. Using first 1000 lines of input files. ---"
TEMP_FASTA_FOR_TEST=$(mktemp)
TEMP_DICT_FOR_TEST=$(mktemp)
head -n 1000 "$FASTA_FILE" > "$TEMP_FASTA_FOR_TEST"
head -n 1000 "$DICT_FILE" > "$TEMP_DICT_FOR_TEST"

FASTA_FILE="$TEMP_FASTA_FOR_TEST"
DICT_FILE="$TEMP_DICT_FOR_TEST"
# --- END OF TEST BLOCK ---

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$UNMAPPED_FILE")"

# Temporary files
TED_IDS_FILE=$(mktemp)
TSV_LOOKUP_FILE=$(mktemp)

# Cleanup function
cleanup() {
    rm -f "$TED_IDS_FILE" "$TSV_LOOKUP_FILE" "${TEMP_FASTA_FOR_TEST:-}" "${TEMP_DICT_FOR_TEST:-}"
}
trap cleanup EXIT

echo "Processing files..."

# Extract TED-IDs from FASTA headers
echo "Extracting TED-IDs from FASTA file..."
grep '^>' "$FASTA_FILE" | sed -e 's/^>//' -e 's/\r$//' > "$TED_IDS_FILE"
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
        # First file (the lookup table): store mappings in an awk associative array.
        # NR==FNR is true only while awk is reading the first file.
        NR==FNR {
            lookup[$1] = $2
            next
        }

        # Initialization block: runs just before processing the second file
        # to correctly print the opening brace for the JSON.
        FNR==1 {
            print "{"
            first_entry = 1
        }

        # Second file (the IDs to map): check against the lookup table.
        {
            if ($1 in lookup) {
                # This ID has a mapping. Handle the comma for subsequent JSON entries.
                if (first_entry == 0) {
                    printf ",\n"
                }
                first_entry = 0

                # Print the JSON key-value pair to stdout.
                printf "  \"%s\": \"%s\"", $1, lookup[$1]
            } else {
                # No mapping found, print the ID to stderr.
                print $1 > "/dev/stderr"
            }
        }

        # END block: runs after all files are processed to close the JSON object.
        END {
            # Add a newline after the last entry if any entries were written.
            if (first_entry == 0) {
                printf "\n"
            }
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
