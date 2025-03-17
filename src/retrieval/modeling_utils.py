from accelerate import Accelerator

import os
import logging

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

def log_message(message, level=logging.WARNING):
    if MAIN_PROCESS:
        print(message)
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