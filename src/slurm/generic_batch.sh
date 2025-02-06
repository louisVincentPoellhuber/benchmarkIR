#!/bin/bash -l

#SBATCH --partition=rali
#SBATCH --cpus-per-task=2
#SBATCH --verbose
#SBATCH --mem=20G

STORAGE_DIR=/Tmp/lvpoellhuber

echo $STORAGE_DIR

# This should already exist on Octal31
source $STORAGE_DIR/bmir-env/bin/activate

export STORAGE_DIR=$STORAGE_DIR
export COMET_API_KEY=TzEzoqltg1eu3XaFzpKHYuQaD
export COMET_PROJECT_NAME=new-attention
#export CUDA_VISIBLE_DEVICES=0

cd Documents/Masters/benchmarkIR

# Change the pipeline or script to the one you wanna run
#bash src/lm/pipelines/finetune_paramsearch.sh     
set -x
bash src/new-attention/pipelines/$1     

#python src/lm/preprocessing.py --task glue --overwrite True

