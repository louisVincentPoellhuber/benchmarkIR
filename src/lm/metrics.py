import os
import argparse
import pandas as pd

argparser = argparse.ArgumentParser("Computing metrics")
argparser.add_argument('--path')
args = argparser.parse_args()
path = args.path

model_name = os.path.basename(os.path.basename(path))
tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"] 

avg_metrics_path = os.path.join(os.path.dirname(path), "avg_metrics.csv")

if os.path.exists(avg_metrics_path):
    avg_metrics = pd.read_csv(avg_metrics_path, index_col=0)
else:
    avg_metrics = pd.DataFrame(columns=tasks)

finetune_paths = [model for model in os.listdir(path) if os.path.isdir(os.path.join(path, model))]

accuracies = []
tasks = []
for model in finetune_paths:
    task = model.split("_")[1]
    metrics_path = os.path.join(os.path.join(path, model), "metrics.csv")
    model_metrics = pd.read_csv(metrics_path)
    avg_acc = model_metrics["accuracy"].mean()
    
    accuracies.append(avg_acc)
    tasks.append(task)

metrics_df = pd.DataFrame(accuracies, index = tasks, columns = [model_name]).T

print(metrics_df.columns)

avg_metrics = pd.concat([avg_metrics, metrics_df])

#print(avg_metrics)

avg_metrics.to_csv(avg_metrics_path)