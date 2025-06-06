#!/bin/bash
#PBS -N lmcd1_cr-dux4
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=64gb
#PBS -l walltime=36:0:0
#PBS -o $BASE_DIR/logs/$PBS_JOBNAME.o$PBS_JOBID
#PBS -e $BASE_DIR/logs/$PBS_JOBNAME.e$PBS_JOBID

if [ -z "$PROTEIN" ]; then
    echo "ERROR: PROTEIN variable not set. Exiting..."
    exit 1
fi

module load AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20230310

WORKDIR=$VSC_DATA_VO_USER/alphafold/runs/${PBS_JOBID}-${PROTEIN}
mkdir -p "$WORKDIR"

cp -a "$PBS_O_WORKDIR/fastas/190525/${PROTEIN}.fasta" "$WORKDIR/"
cd $WORKDIR

echo "Running $PROTEIN.fasta at $(date), output at $WORKDIR"
alphafold --fasta_paths=$PROTEIN.fasta \
          --max_template_date=2999-01-01 \
          --db_preset=full_dbs \
          --output_dir=$PWD \
          --model_preset=multimer
	  --num_runs_per_model=5  || { echo "AlphaFold failed for $PROTEIN"; exit 1; }
	  
echo "Finished $PROTEIN at $(date)"
