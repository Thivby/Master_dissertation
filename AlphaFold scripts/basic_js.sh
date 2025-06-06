#!/bin/bash
#PBS -N dux4
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=64gb
#PBS -l walltime=72:0:0

PROTEIN=dux4

module load AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20230310

WORKDIR=$VSC_DATA_VO_USER/alphafold/runs/$PBS_JOBID-$PROTEIN
mkdir -p $WORKDIR
cp -a $PBS_O_WORKDIR/fastas/$PROTEIN.fasta $WORKDIR/
cd $WORKDIR

echo Running $PROTEIN.fasta, output found at $WORKDIR
alphafold --fasta_paths=$PROTEIN.fasta \
          --max_template_date=2999-01-01 \
          --db_preset=full_dbs \
          --output_dir=$PWD \
          --model_preset=monomer_ptm \
	  --num_multimer_predictions_per_model=5



