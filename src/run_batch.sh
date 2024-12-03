# Update LM files to arcade

rsync -avz --update /home/lvpoellhuber/projects/benchmarkIR/src poellhul@arcade.iro.umontreal.ca:~/Documents/Masters/benchmarkIR-slurm

# Run my script
script=generic_batch.sh


if [[ "$1" == "octal30" || "$1" == "octal31" ]]; then
    gpu="rtx_a5000"
elif [[ "$1" == "octal40" ]]; then
    gpu="ls40"
else
    echo "Unknown computer: $1."
    exit 1  # Exit with an error if the computer is not recognized
fi

echo Run batch. Computer: $1, GPU: $gpu


# Sent to SSH
ssh poellhul@arcade.iro.umontreal.ca "cd ~/Documents/Masters/benchmarkIR-slurm; sbatch --gres=gpu:"$gpu":4 --nodelist="$1" src/slurm/"$script

