import comet_ml

from preprocessing import *
from model_custom_roberta import *
from finetune_roberta import log_message

import argparse
import pandas as pd
import json

from transformers import RobertaConfig, get_scheduler
from roberta_config import CustomRobertaConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from datasets import load_dataset, load_metric
import copy

import dotenv
dotenv.load_dotenv()


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

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)
logging.debug(f"Saving to {STORAGE_DIR}.")



def parse_arguments():
    argparser = argparse.ArgumentParser("Evaluating Roberta")
    argparser.add_argument('--config', default="eval") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) 
    args = argparser.parse_args()

    return args

def main(arg_dict):
    
    task_num_labels = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "qnli": 2,
        "qqp": 2,
        "rte": 2, 
        "sst2": 2,
        "wnli": 2, 
        "stsb": 1
    }
    
    config_dict = arg_dict["config"]
    settings = arg_dict["settings"]
    
    enable_accelerate = settings["accelerate"]
    enable_logging = settings["logging"]

    # Main arguments
    dataset_path = settings["dataset"] 
    model_path = settings["save_path"] # THIS IS NOT A MISTAKE. 
    model_save_path = settings["save_path"]
    tokenizer_path = settings["tokenizer"]

    task = settings["task"]
    num_labels = task_num_labels[task]

    accelerator = Accelerator(log_with="comet_ml", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device 
    metric = load_metric("glue", task, trust_remote_code=True)
    accuracy = load_metric("accuracy", trust_remote_code = True)
    tokenizer = get_tokenizer(tokenizer_path)
    
    test_dataloader = get_dataloader(settings["batch_size"], dataset_path)
    

    config = CustomRobertaConfig.from_dict(config_dict)
    config.vocab_size = tokenizer.vocab_size
    config.num_labels = num_labels

    print(model_path)
    log_message(f"Using the model from: {model_path}", logging.DEBUG, accelerator)
    model = RobertaForSequenceClassification(config=config).from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    model.to(device)

    # Accelerator function
    model, dataloader = accelerator.prepare(
        model, test_dataloader
    )

    loop = tqdm(dataloader, leave=True)
    metrics_df = []
    prediction_distribution = torch.zeros(size=(2, num_labels), device=device)
    unique_labels = torch.arange(num_labels, device=device)
    ones = torch.ones(num_labels, device=device)

    for i, batch in enumerate(loop): 
        input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
        mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
        labels =batch["labels"]#.to(device) 
        
        outputs = model(input_ids, attention_mask = mask, labels = labels)
        predictions = torch.argmax(outputs.logits, axis=1)
        metrics = metric.compute(predictions=predictions, references=labels)

        
        label_counts = torch.cat((labels, unique_labels)).unique(return_counts=True)
        prediction_distribution[0] += label_counts[1] - ones
        pred_counts =  torch.cat((predictions, unique_labels)).unique(return_counts=True)
        prediction_distribution[1] += pred_counts[1] - ones
        
        row = [float(outputs.loss)] + list(metrics.values())
        row_names = ["loss"] + list(metrics.keys())

        if "accuracy" not in row_names:
            train_acc = accuracy.compute(predictions=predictions, references=labels)
            row.append(train_acc["accuracy"])
            row_names.append("accuracy")
        metrics_df.append(row)

    log_message(f"Saving metric data to: {model_save_path}", logging.DEBUG, accelerator)

    print(os.path.join(model_save_path, "metrics.csv"))
    metrics_df = pd.DataFrame(metrics_df, columns = row_names)
    avg_metrics = metrics_df.mean()
    metrics_df.to_csv(os.path.join(model_save_path, "metrics.csv"))
    avg_metrics.to_csv(os.path.join(model_save_path, "avg_metrics.csv"))

    print(f"Average accuracy: {avg_metrics.loc['accuracy']}.")
    log_message(f"Average accuracy: {avg_metrics.loc['accuracy']}.", logging.WARNING, accelerator)
    
    dist_path = os.path.join(os.path.dirname(model_save_path), "prediction_dist.csv")
    dist_df = pd.DataFrame(prediction_distribution.cpu().numpy()).T
    dist_df = dist_df[[1, 0]]
    dist_df.index = [f"{task}_{i}" for i in range(num_labels)]
    dist_df.columns = ["predictions", "labels"]

    print(dist_df)
    if os.path.exists(dist_path):
        total_dist_df = pd.read_csv(dist_path, index_col=0)
        dist_df = pd.concat([total_dist_df, dist_df])
    dist_df.to_csv(dist_path)


if __name__ == "__main__":
    args = parse_arguments()

    if len(args.config_dict)>0:
        original_arg_dict = json.loads(args.config_dict)
    else:   
        config_dict = os.path.join("src/lm/configs", args.config+"_finetune.json")
        with open(config_dict) as fp: 
            original_arg_dict = json.load(fp)
    
    for key in original_arg_dict["settings"]:
        if type(original_arg_dict["settings"][key]) == str:
            original_arg_dict["settings"][key] = original_arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)
    
    log_message(f"============ Evaluating {original_arg_dict['settings']['exp_name']}. ============", logging.WARNING, None)
    log_message(f"Model Configuration: {original_arg_dict}", logging.INFO, None)

    if original_arg_dict["settings"]["task"]=="glue":
        #tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]        
        tasks = ["qnli"]
        #tasks = ["cola", "mrpc", "rte", "wnli"]

        for task in tasks:
            print(f"============ Processing {task} ============")
            log_message(f"Processing {task}.", logging.WARNING, None)

            # Adjusting the config for each task
            arg_dict = copy.deepcopy(original_arg_dict)

            model_name = arg_dict["settings"]["exp_name"].split("_")[0]

            if arg_dict["settings"]["model"] != "FacebookAI/roberta-base":
                arg_dict["settings"]["model"] = os.path.join(arg_dict["settings"]["model"], "roberta_"+task)
            else:
                for temp_task in tasks:
                    save_path = os.path.join(arg_dict["settings"]["save_path"], "roberta_"+temp_task)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

            arg_dict["settings"]["task"] = task
            arg_dict["settings"]["save_path"] = os.path.join(arg_dict["settings"]["save_path"], "roberta_"+task)  # Changed 'save_path' from 'model'
            
            arg_dict["settings"]["dataset"] = os.path.join(arg_dict["settings"]["dataset"], os.path.join(task, task+"_test.pt"))

            main(arg_dict)
        
    else:
        main(original_arg_dict)