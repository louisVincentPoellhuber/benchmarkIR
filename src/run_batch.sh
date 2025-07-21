# Update LM files to arcade

rsync -avz --update /home/lvpoellhuber/projects/benchmarkIR/src arcade:~/Documents/Masters/benchmarkIR-slurm
# rsync -avz --update  arcade:~/Documents/Masters/benchmarkIR-slurm/src /home/lvpoellhuber/projects/benchmarkIR

# Run my script
script=generic_batch.sh

specific_node=True
if [[ "$1" == "octal30" ]]; then
    gpu="rtx_a5000"
    part="02"
elif [[ "$1" == "octal31" ]]; then
    gpu="rtx_a5000"
    part="01"
elif [[ "$1" == "octal35" ]]; then
    gpu="rtx_a6000"
    part="01"
elif [[ "$1" == "octal40" || "$1" == "octal41"  || "$1" == "ls40" ]]; then
    gpu="ls40"
    part="01"
    specific_node=False
else
    echo "Unknown computer: $1."
    exit 1  # Exit with an error if the computer is not recognized
fi


# Sent to SSH
if [[ "$specific_node" == True ]]; then
    echo "
    |
    | Running batch: $2.
    |
    | Computer: $1
    | GPU: $gpu
    | Storage part: $part
    |"
    ssh arcade "cd ~/Documents/Masters/benchmarkIR-slurm; sbatch --gres=gpu:"$gpu":4 --nodelist="$1" src/slurm/"$script" "$2" "$part
else
    echo "
    |
    | Running batch: $2.
    |
    | Computer: Any
    | GPU: $gpu
    | Storage part: $part
    |"
    ssh arcade "cd ~/Documents/Masters/benchmarkIR-slurm; sbatch --gres=gpu:"$gpu":4 src/slurm/"$script" "$2" "$part
fi

