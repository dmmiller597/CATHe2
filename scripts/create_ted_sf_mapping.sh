#!/usr/bin/env bash
# david miller - 16/06/2025
# Create TED-ID → CATH-SF mapping from FASTA & TSV files in a single awk pass.
# Designed for large datasets (millions of sequences) – minimal disk I/O, robust, and fully POSIX-compliant.

set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Default locations – override with CLI flags if required
# -----------------------------------------------------------------------------
FASTA_FILE="/SAN/orengolab/cath_alphafold/cath_ted_gold_sequences_hmmvalidated_qscore_0.7_S100_rep_seq.fasta"
DICT_FILE="/state/partition2/NO_BACKUP/databases/ted/datasets/ted_365m.domain_summary.cath.globularity.taxid.qscore.tsv"
OUTPUT_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-mapping.json"
UNMAPPED_FILE="/SAN/orengolab/functional-families/CATHe2/data/TED/TED-SF-unmapped.txt"

usage() {
  cat <<EOF
Usage: ${0##*/} [-f FASTA_FILE] [-d TSV_DICT] [-o OUTPUT_JSON] [-u UNMAPPED_IDS]

Maps TED identifiers in FASTA_FILE to their CATH-SF annotations in TSV_DICT.
Outputs:
  * OUTPUT_JSON   – JSON dictionary TED_ID → CATH-SF
  * UNMAPPED_IDS  – newline-separated list of TED_IDs without mapping
EOF
  exit 1
}

# -----------------------------------------------------------------------------
# CLI parsing ──────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
while getopts ":f:d:o:u:h" opt; do
  case "$opt" in
    f) FASTA_FILE=$OPTARG ;;
    d) DICT_FILE=$OPTARG ;;
    o) OUTPUT_FILE=$OPTARG ;;
    u) UNMAPPED_FILE=$OPTARG ;;
    h|\?) usage ;;
  esac
done

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
for f in "$FASTA_FILE" "$DICT_FILE"; do
  if [[ ! -r "$f" ]]; then
    echo "ERROR: Cannot read file: $f" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$UNMAPPED_FILE")"

# -----------------------------------------------------------------------------
# Mapping (single awk pass) ────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
# 1. Read TSV dictionary first (ARGIND 1) into associative array dict[ted_id] = cath_sf
# 2. Scan FASTA headers, perform lookup, build JSON, and write unmapped IDs.
#
#   * Handles duplicate IDs gracefully (processed once).
#   * Works with CRLF-terminated files.
#   * Emits summary statistics to stderr.

awk -v json="$OUTPUT_FILE" -v unmapped="$UNMAPPED_FILE" '
  BEGIN {
    FS="\t"; ORS="";
    mapped=0; total=0; unmapped_cnt=0;
    print "{\n" > json           # open JSON file
  }

  # ---------------- TSV dictionary ----------------
  FNR==NR {
    if (NF >= 15 && $15 != "-" && $15 != "")
      dict[$1] = $15
    next
  }

  # ---------------- FASTA processing -------------
  /^>/ {
    id = substr($0, 2)
    sub(/\r$/, "", id)
    if (id == "") next
    if (++seen[id] == 1) {
      total++
      if (id in dict) {
        if (mapped++) print ",\n" >> json
        val = dict[id]
        gsub(/"/, "\\\"", val)
        printf "  \"%s\": \"%s\"", id, val >> json
      } else {
        print id >> unmapped
        unmapped_cnt++
      }
    }
    next
  }

  END {
    print "\n}\n" >> json        # close JSON
    cov = (total ? (mapped / total) * 100 : 0)
    printf "Total IDs: %d\nMapped: %d\nUnmapped: %d\nCoverage: %.2f%%\n", total, mapped, unmapped_cnt, cov > "/dev/stderr"
  }
' "$DICT_FILE" "$FASTA_FILE"

# -----------------------------------------------------------------------------
# Epilogue
# -----------------------------------------------------------------------------

echo "JSON mapping written to: $OUTPUT_FILE"
echo "Unmapped IDs written to: $UNMAPPED_FILE"
echo "Done."
