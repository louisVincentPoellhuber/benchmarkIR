# Update LM files to arcade

rsync -avz --update /home/lvpoellhuber/projects/benchmarkIR/src poellhul@arcade.iro.umontreal.ca:~/Documents/Masters/benchmarkIR-slurm

# Run my script
script=generic_batch.sh
# Sent to SSH
ssh poellhul@arcade.iro.umontreal.ca "cd ~/Documents/Masters/benchmarkIR-slurm; sbatch src/slurm/"$script


# Sync the new files with the old ones
#rsync -avz --update poellhul@arcade.iro.umontreal.ca:~/Tmp/lvpoellhuber/models /home/lvpoellhuber/storage/models
