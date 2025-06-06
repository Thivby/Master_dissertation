#!/bin/bash

BASE_DIR="$VSC_DATA_VO_USER/alphafold"
FASTAS_DIR="$BASE_DIR/fastas/190525"
JOB_SCRIPT="$BASE_DIR/my_js.sh"

for fasta in "$FASTAS_DIR"/*.fasta; do
    PROTEIN=$(basename "$fasta" .fasta)
    
    echo "Submitting job for $PROTEIN"
    qsub -v PROTEIN="$PROTEIN" "$JOB_SCRIPT"
done 
