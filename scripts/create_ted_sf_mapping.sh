#!/bin/bash
# david miller - 16/06/2025
# Create TED-ID to CATH-SF mapping from FASTA and TSV files
# need to run on bubba213-3 

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
TED_IDS_FILE=$(mktemp)
TSV_LOOKUP_FILE=$(mktemp)

# Cleanup function
cleanup() {
    rm -f "$TED_IDS_FILE" "$TSV_LOOKUP_FILE"
}
trap cleanup EXIT

echo "Processing files..."

# Extract TED-IDs from FASTA headers
echo "Extracting TED-IDs from FASTA file..."
grep '^>' "$FASTA_FILE" | sed 's/^>//' > "$TED_IDS_FILE"
TOTAL_FASTA_IDS=$(wc -l < "$TED_IDS_FILE")
echo "Found $TOTAL_FASTA_IDS TED-IDs in FASTA file"

# Create lookup table from TSV (TED-ID -> CATH-SF)
echo "Creating lookup table from TSV file..."
awk -F '\t' 'NF >= 15 && $15 != "-" && $15 != "" { print $1 "\t" $15 }' "$DICT_FILE" > "$TSV_LOOKUP_FILE"
TOTAL_TSV_MAPPINGS=$(wc -l < "$TSV_LOOKUP_FILE")
echo "Found $TOTAL_TSV_MAPPINGS TED-IDs with CATH-SF mappings in TSV file"

# Load TSV data into associative array
echo "Loading TSV mappings..."
declare -A tsv_lookup
while IFS=$'\t' read -r ted_id cath_sf; do
    tsv_lookup["$ted_id"]="$cath_sf"
done < "$TSV_LOOKUP_FILE"

# Process FASTA TED-IDs and create mapping
echo "Creating JSON mapping..."
declare -a unmapped_ids
mapped_count=0
first_entry=true

# Create JSON file
{
    echo "{"
    
    while IFS= read -r ted_id; do
        if [[ -n "${tsv_lookup[$ted_id]}" ]]; then
            # Found mapping
            if [[ "$first_entry" == "true" ]]; then
                first_entry=false
            else
                echo ","
            fi
            printf "  \"%s\": \"%s\"" "$ted_id" "${tsv_lookup[$ted_id]}"
            ((mapped_count++))
        else
            # No mapping found
            unmapped_ids+=("$ted_id")
        fi
    done < "$TED_IDS_FILE"
    
    echo ""
    echo "}"
} > "$OUTPUT_FILE"

# Write unmapped IDs to separate file
if [[ ${#unmapped_ids[@]} -gt 0 ]]; then
    printf '%s\n' "${unmapped_ids[@]}" > "$UNMAPPED_FILE"
    echo "Unmapped IDs written to: $UNMAPPED_FILE"
else
    # Create empty unmapped file
    touch "$UNMAPPED_FILE"
    echo "All TED-IDs were successfully mapped!"
fi

# Display summary
echo ""
echo "=== SUMMARY ==="
echo "Total FASTA TED-IDs: $TOTAL_FASTA_IDS"
echo "Successfully mapped: $mapped_count"
echo "Unmapped: ${#unmapped_ids[@]}"
if [[ $TOTAL_FASTA_IDS -gt 0 ]]; then
    echo "Mapping coverage: $(( mapped_count * 100 / TOTAL_FASTA_IDS ))%"
fi
echo ""
echo "JSON mapping written to: $OUTPUT_FILE"
echo "Script completed successfully!"
