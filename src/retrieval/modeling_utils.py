from accelerate import Accelerator

import os
import logging

from transformers import AutoConfig, AutoModel
from model_longtriever import LongtrieverConfig, Longtriever, HierarchicalLongtrieverConfig, HierarchicalLongtriever

AutoConfig.register("longtriever", LongtrieverConfig)
AutoModel.register(LongtrieverConfig, Longtriever)
AutoConfig.register("hierarchical_longtriever", HierarchicalLongtrieverConfig)
AutoModel.register(HierarchicalLongtrieverConfig, HierarchicalLongtriever)

import dotenv
dotenv.load_dotenv()


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

MAIN_PROCESS = Accelerator().is_main_process
STORAGE_DIR = os.getenv("STORAGE_DIR")

def log_message(message, level=logging.WARNING, force_message = False):
    if force_message or MAIN_PROCESS:
        # print(message)
        logging.log(msg=message, level=level)


def log_metrics(q_model, doc_model, scheduler, optim, experiment, loss, step):
    # Loss
    experiment.log_metrics({"loss": loss}, step=step)

    # Learning rate
    current_lr = scheduler.get_lr()[0]
    experiment.log_metric("lr", current_lr, step=step)

    # Gradient norm
    total_norm = 0
    for name, param in q_model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    
        #     param_grad = param.grad.abs().mean().item()
        #     if param_grad < 1e-5:
        #         log_message(f"{name} has small grad {param_grad}", logging.DEBUG)
        # else:
        #     log_message(f"{name} has no grad", logging.DEBUG)
        
    total_norm = total_norm ** 0.5
    experiment.log_metric("q_total_grad_norm", total_norm, step=step)

    total_norm = 0
    for name, param in doc_model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
        #     param_grad = param.grad.abs().mean().item()
        #     if param_grad < 1e-5:
        #         log_message(f"{name} has small grad {param_grad}", logging.DEBUG)
        # else:
        #     log_message(f"{name} has no grad", logging.DEBUG)
    
    total_norm = total_norm ** 0.5
    experiment.log_metric("doc_total_grad_norm", total_norm, step=step)


def default_args(arg_dict):
    settings = arg_dict["settings"]
    keys = settings.keys()

    if "model" not in keys: settings["model"] = "FacebookAI/roberta-base"
    if "save_path" not in keys: settings["save_path"] = "STORAGE_DIR/models"
    if "tokenizer" not in keys: settings["tokenizer"] = "google-bert/bert-base-uncased"
    if "accelerate" not in keys: settings["accelerate"] = True
    if "logging" not in keys: settings["logging"] = True
    if "epochs" not in keys: settings["epochs"] = 10
    if "batch_size" not in keys: settings["batch_size"] = 3
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
    if "save_total_limit" not in keys: settings["save_total_limit"] = 5 # Useful for if the code crashes: restart from last epoch
    if "optim" not in keys: settings["optim"] = "adamw_torch"
    if "resume_from_checkpoint" not in keys: settings["resume_from_checkpoint"] = None 
    if "logging_steps" not in keys: settings["logging_steps"] = 10

    if ("task" not in keys) | ("exp_name" not in keys):
        raise Exception("Experiment not set up. Please provide an experiment name and a task.") 

    arg_dict["settings"] = settings
    
    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)


    return arg_dict