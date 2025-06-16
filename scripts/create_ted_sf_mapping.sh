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

# -------------------------------------------------------------------
# Memory-only mapping: read FASTA first to collect required TED IDs, then
# stream through the massive TSV.  No temporary files or external sort.
# -------------------------------------------------------------------

awk -F '\t' \
    -v unmapped_file="$UNMAPPED_FILE" \
    -v output_json_path="$OUTPUT_JSON" \
'
# -------------------- PASS 1: FASTA -------------------------------
FNR==NR {
    if ($0 ~ /^>/) {
        id = substr($1, 2)            # strip leading ">"
        if (!(id in needed)) {
            needed[id] = 1
            total++
        }
    }
    next                               # continue reading FASTA only
}

# -------------------- PASS 2: TSV ---------------------------------
NF >= 15 {
    id = $1; sf = $15
    if (id in needed && sf != "" && sf != "-") {
        if (!printed[id]) {
            printed[id] = 1
            if (count) printf ",\n"
            printf "  \"%s\": \"%s\"", id, sf
            delete needed[id]
            count++
        }
    }
}

# -------------------- END -----------------------------------------
END {
    if (count) printf "\n"
    print "}"

    for (id in needed) print id >> unmapped_file
    close(unmapped_file)

    unmapped = total - count
    cov = (total ? count / total * 100 : 0)

    printf "\nSummary\n-------\n" > "/dev/stderr"
    printf "Total TED IDs:    %d\n", total        > "/dev/stderr"
    printf "Mapped:           %d\n", count        > "/dev/stderr"
    printf "Unmapped:         %d\n", unmapped     > "/dev/stderr"
    printf "Coverage:         %.1f%%\n", cov         > "/dev/stderr"
    printf "-----------------\n"               > "/dev/stderr"
    printf "Output JSON:      %s\n", output_json_path > "/dev/stderr"
    printf "Unmapped IDs:     %s\n", unmapped_file     > "/dev/stderr"
}
' "$FASTA_FILE" "$DICT_FILE" > "$OUTPUT_JSON"

echo "Done." >&2