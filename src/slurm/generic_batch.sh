#!/bin/bash -l

#SBATCH --partition=rali
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx_a5000:4
#SBATCH --nodelist=octal30


STORAGE_DIR=/Tmp/lvpoellhuber

echo $STORAGE_DIR

# This should already exist on Octal31
source $STORAGE_DIR/bmir-env/bin/activate

export STORAGE_DIR=$STORAGE_DIR

cd Documents/Masters/benchmarkIR

# Change the pipeline or script to the one you wanna run
bash src/lm/pipelines/finetune_optim.sh     

#python src/lm/preprocessing.py --task glue --overwrite True
