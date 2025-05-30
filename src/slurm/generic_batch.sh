#!/bin/bash -l

#SBATCH --partition=rali
#SBATCH --verbose
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G

STORAGE_DIR=/part/$2/Tmp/lvpoellhuber

echo $STORAGE_DIR

# This should already exist on Octal31
source $STORAGE_DIR/bmir-env/bin/activate

# pip install -r src/requirements.txt

export STORAGE_DIR=$STORAGE_DIR
export COMET_API_KEY=TzEzoqltg1eu3XaFzpKHYuQaD
export COMET_PROJECT_NAME=new-attention
#export CUDA_VISIBLE_DEVICES=0

cd Documents/Masters/benchmarkIR

# Change the pipeline or script to the one you wanna runs
set -x
# bash src/retrieval/pipelines/$1     
bash src/longtriever/pipelines/$1     


