import os
import argparse
import pandas as pd
import datetime
import json

argparser = argparse.ArgumentParser("Computing metrics")
argparser.add_argument('--path')
argparser.add_argument('--config')
args = argparser.parse_args()
path = args.path
config = json.loads(args.config)

# Model name
model_name = os.path.basename(os.path.basename(path))


# Find models
all_tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"] 
experiment_df_path = os.path.join(os.path.dirname(path), "experiment_df.csv")
if os.path.exists(experiment_df_path):
    experiment_df = pd.read_csv(experiment_df_path, index_col=0)
else:
    experiment_df = pd.DataFrame(columns=["date", "computer", "job_id"] + all_tasks + ["model_path", "config"])

finetune_paths = [model for model in os.listdir(path) if os.path.isdir(os.path.join(path, model))]

# Adding stuff to the row
row = []
columns = []

# Time
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
row.append(now)
columns.append("date")

# Slurm info
job_computer = os.getenv("SLURM_NODELIST")
if job_computer == None: job_computer = "local"
row.append(job_computer)
columns.append("computer")

job_id = os.getenv("SLURM_JOB_ID")
if job_id == None: job_id = "local"
row.append(job_id)
columns.append("job_id")


# Computing metrics
for model in finetune_paths:
    task = model.split("_")[1]
    metrics_path = os.path.join(os.path.join(path, model), "metrics.csv")

    if os.path.exists(metrics_path):
        model_metrics = pd.read_csv(metrics_path)
        avg_acc = model_metrics["accuracy"].mean()
        
        row.append(avg_acc)
        columns.append(task)

# Model info
row.append(path)
columns.append("model_path")

row.append(config)
columns.append("config")


metrics_df = pd.DataFrame(row, index = columns, columns = [model_name]).T

print(metrics_df.columns)

experiment_df = pd.concat([experiment_df, metrics_df])
    
experiment_df.to_csv(experiment_df_path)