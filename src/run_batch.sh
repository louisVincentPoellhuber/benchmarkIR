# Update LM files to arcade

rsync -avz --update /home/lvpoellhuber/projects/benchmarkIR/src arcade:~/Documents/Masters/benchmarkIR-slurm
# rsync -avz --update  arcade:~/Documents/Masters/benchmarkIR-slurm/src /home/lvpoellhuber/projects/benchmarkIR

# Run my script
script=generic_batch.sh


if [[ "$1" == "octal30" ]]; then
    gpu="rtx_a5000"
    part="02"
elif [[ "$1" == "octal31" ]]; then
    gpu="rtx_a5000"
    part="01"
elif [[ "$1" == "octal40" || "$1" == "octal41" ]]; then
    gpu="ls40"
    part="01"
else
    echo "Unknown computer: $1."
    exit 1  # Exit with an error if the computer is not recognized
fi

echo "
|
| Running batch: $2.
|
| Computer: $1
| GPU: $gpu
| Storage part: $part
|"


# Sent to SSH
ssh arcade "cd ~/Documents/Masters/benchmarkIR-slurm; sbatch --gres=gpu:"$gpu":4 --nodelist="$1" src/slurm/"$script" "$2" "$part

