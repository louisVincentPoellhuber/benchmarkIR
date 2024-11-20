#!/bin/bash -l

#SBATCH --partition=rali
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --nodelist=octal31


STORAGE_DIR=/Tmp/lvpoellhuber

echo $STORAGE_DIR

# This should already exist on Octal31
source $STORAGE_DIR/bmir-env/bin/activate

#pip install python-dotenv # to remove

export STORAGE_DIR=$STORAGE_DIR

cd Documents/Masters/benchmarkIR

# Everything should run normally, saving everything to the STORAGE_DIR
bash src/lm/pipelines/baseline_eval.sh

