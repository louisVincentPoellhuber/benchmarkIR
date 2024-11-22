#!/bin/bash -l

#SBATCH --partition=rali
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ls40:4
#SBATCH --nodelist=octal40


STORAGE_DIR=/Tmp/lvpoellhuber

echo $STORAGE_DIR

# This should already exist on Octal31
source $STORAGE_DIR/bmir-env/bin/activate

export STORAGE_DIR=$STORAGE_DIR
export COMET_API_KEY=TzEzoqltg1eu3XaFzpKHYuQaD


cd Documents/Masters/benchmarkIR

# Change the pipeline or script to the one you wanna run
bash src/lm/pipelines/finetune_optim.sh     

#python src/lm/preprocessing.py --task glue --overwrite True

