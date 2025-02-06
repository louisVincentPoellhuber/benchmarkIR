import os
import argparse
import pandas as pd
import datetime
import json

from finetune_roberta import log_message

import logging
JOBID = os.getenv("SLURM_JOB_ID")
if JOBID == None: JOBID = "local"
logging.basicConfig( 
    encoding="utf-8", 
    filename=f"slurm-{JOBID}.log", 
    filemode="a", 
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
    )


def parse_arguments():
    argparser = argparse.ArgumentParser("Computing metrics")
    argparser.add_argument('--path')
    argparser.add_argument('--config_dict')
    args = argparser.parse_args()

    return args


def compute_metrics(save_path, config):
    print("Computing metrics.")
    
    # Model name
    model_name = os.path.basename(os.path.basename(save_path))

    # Find models
    all_tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"] 
    experiment_df_path = os.path.join(os.path.dirname(os.path.dirname(save_path)), "experiment_df.csv")
    if os.path.exists(experiment_df_path):
        experiment_df = pd.read_csv(experiment_df_path, index_col=0)
    else:
        experiment_df = pd.DataFrame(columns=["date", "computer", "job_id"] + all_tasks + ["model_path", "config"])

    finetune_paths = [model for model in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, model))]
    print(save_path)
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
    tasks = []
    for model in finetune_paths:
        task = model.split("_")[1]
        metrics_path = os.path.join(os.path.join(save_path, model), "avg_metrics.csv")

        if os.path.exists(metrics_path):
            model_metrics = pd.read_csv(metrics_path)
            avg_acc = model_metrics.loc["accuracy"]
            
            row.append(avg_acc)
            columns.append(task)
            tasks.append(task)


    # Model info
    row.append(save_path)
    columns.append("model_path")

    row.append(config)
    columns.append("config")


    metrics_df = pd.DataFrame(row, index = columns, columns = [model_name]).T

    print(metrics_df[tasks])

    experiment_df = pd.concat([experiment_df, metrics_df])
        
    experiment_df.to_csv(experiment_df_path)

if __name__ == "__main__":
    args = parse_arguments()
    
    save_path = args.path
    config = json.loads(args.config_dict)
    log_message(f"============ Metrics for {config['settings']['exp_name']}. ============", logging.WARNING, None)

    compute_metrics(save_path, config)