#!/bin/bash -l

#SBATCH --partition=rali
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --nodelist=octal30


STORAGE_DIR=/Tmp/lvpoellhuber

echo $STORAGE_DIR
#pip install python-dotenv # to remove

export STORAGE_DIR=$STORAGE_DIR

cd
cd ..
cd ..
cd Tmp/lvpoellhuber/models
ls

rsync -avz --update --progress finetune_roberta poellhul@octal30:/Tmp/lvpoellhuber/models/finetune_roberta

