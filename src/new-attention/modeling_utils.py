import comet_ml

from model_custom_roberta import *

import argparse
import json
import datetime
import numpy as np

from accelerate import Accelerator
import os
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

import dotenv
dotenv.load_dotenv()


MAIN_PROCESS = Accelerator().is_main_process

STORAGE_DIR = os.getenv("STORAGE_DIR")
logging.debug(f"Saving to {STORAGE_DIR}.")

def log_message(message, level=logging.WARNING):
    if MAIN_PROCESS:
        print(message)
        logging.log(msg=message, level=level)

def parse_arguments():
    argparser = argparse.ArgumentParser("Argument Parser")
    argparser.add_argument('--config', default="test.json") 
    argparser.add_argument('--config_dict', default={}) 
    args = argparser.parse_args()

    return args

def log_metrics(model, scheduler, optim, experiment, loss, step):
    # Loss
    experiment.log_metrics({"loss": loss}, step=step)

    # Learning rate
    current_lr = scheduler.get_lr()
    experiment.log_metric("lr", current_lr, step=step)

    # Gradient norm
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    
    total_norm = total_norm ** 0.5
    experiment.log_metric("total_grad_norm", total_norm, step=step)
        

def default_args(arg_dict):
    settings = arg_dict["settings"]
    keys = settings.keys()

    if "model" not in keys: settings["model"] = "FacebookAI/roberta-base"
    if "save_path" not in keys: settings["save_path"] = "STORAGE_DIR/models"
    if "dataset_dir" not in keys: settings["dataset_dir"] = "STORAGE_DIR/datasets"
    if "tokenizer" not in keys: settings["tokenizer"] = "FacebookAI/roberta-base"
    if "train" not in keys: settings["train"] = True
    if "validate" not in keys: settings["validation"] = False
    if "evaluate" not in keys: settings["eval"] = False
    if "eval_strategy" not in keys: settings["eval_strategy"] = "no"
    if "accelerate" not in keys: settings["accelerate"] = True
    if "logging" not in keys: settings["logging"] = True
    if "epochs" not in keys: settings["epochs"] = 10
    if "batch_size" not in keys: settings["batch_size"] = 32
    if "lr" not in keys: settings["lr"] = 1e-4
    if "weight_decay" not in keys: settings["weight_decay"] = 0.01
    if "adam_beta1" not in keys: settings["adam_beta1"] = 0.9
    if "adam_beta2" not in keys: settings["adam_beta2"] = 0.999
    if "adam_epsilon" not in keys: settings["adam_epsilon"] = 1e-8
    if "lr_scheduler_type" not in keys: settings["lr_scheduler_type"] = "linear"
    if "warmup_ratio" in keys: 
        settings["warmup_steps"] = 0
    elif "warmup_steps" in keys:
        settings["warmup_ratio"] = 0.0
    else:
        settings["warmup_steps"] = 0
        settings["warmup_ratio"] = 0.06
    if "save_strategy" not in keys: settings["save_strategy"] = "epoch"
    if "save_total_limit" not in keys: settings["save_total_limit"] = 1 # Useful for if the code crashes: restart from last epoch
    if "optim" not in keys: settings["optim"] = "adamw_torch"
    if "auto_find_batch_size" not in keys: settings["auto_find_batch_size"] = False 
    if "resume_from_checkpoint" not in keys: settings["resume_from_checkpoint"] = None 
    if "logging_steps" not in keys: settings["logging_steps"] = 10

    if ("task" not in keys) | ("exp_name" not in keys):
        raise Exception("Experiment not set up. Please provide an experiment name and a task.") 

    arg_dict["settings"] = settings
    
    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)


    return arg_dict


def compute_metrics(eval_output, arg_dict):
    # Mod
    experiment_info = {}

    exp_name = arg_dict["settings"]["exp_name"]
    task = arg_dict["settings"]["task"]

    # Time
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_info["date"] = now

    # Slurm info
    job_computer = os.getenv("SLURM_NODELIST")
    if job_computer == None: job_computer = "local"
    experiment_info["computer"] = job_computer

    job_id = os.getenv("SLURM_JOB_ID")
    if job_id == None: job_id = "local"
    experiment_info["job_id"] = job_id

    # Computing metrics
    metrics = eval_output
    unwanted_keys = ["eval_loss", "eval_model_preparation_time", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"]
    for key in unwanted_keys: 
        if key in metrics.keys():
            metrics.pop(key)
    metrics = {(key[len("eval_"):]) if key.startswith("eval_") else key: value for key, value in metrics.items()}
    
    experiment_info["metrics"] = metrics
    log_message(f"Evaluation metrics: {metrics}", logging.WARNING)
    
    #/storage/models/main_branch/model_type/experiment/dataset/...
    experiments_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(arg_dict["settings"]["save_path"]))), "experiments.json")

    if not os.path.exists(experiments_path):
        with open(experiments_path, "a+") as fp:
            experiment_info = {exp_name:experiment_info}
            json.dump(experiment_info, fp, indent=4)
    else: 
        # Load the experiments
        with open(experiments_path, "r") as fp:
            experiments = json.load(fp)

        # If the experiment already exists, append the metrics to its item
        if exp_name in experiments.keys():
            # Seek a new number
            i = 0
            while task in experiments[exp_name].keys():
                i += 1
                task = f"{arg_dict['settings']['task']}{i}"

            experiment_info = {task:experiment_info}
            experiments[exp_name] = experiments[exp_name] | experiment_info
        else: # If not, create it
            experiment_info = {task:experiment_info}
            experiment_info = {exp_name:experiment_info}
            experiments = experiments | experiment_info
        
        # Overwrite the file
        with open(experiments_path, "w") as fp:
            json.dump(experiments, fp, indent=4)


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    
    def __getitem__(self, i):
        return  {key: tensor[i] for key, tensor in self.encodings.items()}
    
    def save(self, save_path):
        torch.save(self.encodings, save_path)
    
