from accelerate import Accelerator

import os
import logging
import torch
import time
import pytrec_eval
from beir.retrieval.evaluation import EvaluateRetrieval


from transformers import AutoConfig, AutoModel
from model_longtriever import LongtrieverConfig, Longtriever, HierarchicalLongtrieverConfig, HierarchicalLongtriever

AutoConfig.register("longtriever", LongtrieverConfig)
AutoModel.register(LongtrieverConfig, Longtriever)
AutoConfig.register("hierarchical_longtriever", HierarchicalLongtrieverConfig)
AutoModel.register(HierarchicalLongtrieverConfig, HierarchicalLongtriever)

import dotenv
dotenv.load_dotenv()

from losses import *

LOSS_FUNCTIONS = {
    "cross_entropy": InBatchNegativeLoss(),
    "triplet": TripletLoss(),
}


JOBID = os.getenv("SLURM_JOB_ID")
if JOBID == None: # No slurm? Try to get the experiment name exported in a bash file
    JOBID = os.getenv("EXP_NAME")
    if JOBID == None: # No experiment name? Use local
        JOBID = "local"
    os.makedirs(f"src/retrieval/logs", exist_ok=True)
    log_path = f"src/retrieval/logs/{JOBID}.log"
else:
    log_path = f"slurm-{JOBID}.log"

# if JOBID == None: JOBID = "debug"
logging.basicConfig( 
    encoding="utf-8", 
    filename=log_path, 
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


def scan_gradient_norm(biencoder, step, batch, path):
     # Calculate gradient norms
    query_norm = torch.norm(torch.stack([
        p.grad.detach().norm(2) for p in biencoder.q_model.parameters() if p.grad is not None
    ]), 2).item()

    doc_norm = torch.norm(torch.stack([
        p.grad.detach().norm(2) for p in biencoder.doc_model.parameters() if p.grad is not None
    ]), 2).item()

    # Threshold to catch extreme gradients
    if query_norm > 50.0 or doc_norm > 50.0:
        log_message(f"Step {step} has high gradient norm. Query: {query_norm:.2f}, Doc: {doc_norm:.2f}", logging.ERROR)
        save_path = os.path.join(path, f"problematic_batch_step_{step}.pt")
        torch.save(batch, save_path)



def default_args(arg_dict):
    settings = arg_dict["settings"]
    keys = settings.keys()

    if "model" not in keys: settings["model"] = "google-bert/bert-base-uncased"
    if "save_path" not in keys: settings["save_path"] = "STORAGE_DIR/models"
    if "tokenizer" not in keys: settings["tokenizer"] = "google-bert/bert-base-uncased"
    if "logging" not in keys: settings["logging"] = True
    if "epochs" not in keys: settings["epochs"] = 3
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
        settings["warmup_ratio"] = 0.1
    if "optim" not in keys: settings["optim"] = "adamw_torch"
    if "resume_from_checkpoint" not in keys: settings["resume_from_checkpoint"] = None 
    if "logging_steps" not in keys: settings["logging_steps"] = 100
    if "checkpoint_steps" not in keys: settings["checkpoint_steps"] = 10000
    if "gradient_accumulation_steps" not in keys: settings["gradient_accumulation_steps"] = 1
    if "seed" not in keys: settings["seed"] = 42
    if "loss_fn" not in keys: 
        settings["loss_fn"] = "cross_entropy"
    elif settings["loss_fn"] not in LOSS_FUNCTIONS.keys():
        log_message(f"Loss function {settings['loss_fn']} not recognized. Using default: cross_entropy.", logging.WARNING)
        settings["loss_fn"] = "cross_entropy"

    if ("task" not in keys) | ("exp_name" not in keys):
        raise Exception("Experiment not set up. Please provide an experiment name and a task.") 

    config_dict = arg_dict["config"]
    keys = config_dict.keys()
    if "attn_implementation" not in keys: config_dict["attn_implementation"] = "eager"
    if "query_prompt" not in keys: config_dict["query_prompt"] = ""
    if "passage_prompt" not in keys: config_dict["passage_prompt"] = ""
    if "sep" not in keys: config_dict["sep"] = " [SEP] "



    arg_dict["settings"] = settings
    arg_dict["config"] = config_dict
    
    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)
    for key in arg_dict["config"]:
        if type(arg_dict["config"][key]) == str:
            arg_dict["config"][key] = arg_dict["config"][key].replace("STORAGE_DIR", STORAGE_DIR)
    

    return arg_dict


class CustomEvaluateRetrieval(EvaluateRetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = True,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logging.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        mrr = {}

        mrr[f"MRR"] = 0.0
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        mrr_string = "recip_rank." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, mrr_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            mrr[f"MRR"] += scores[query_id]["recip_rank"]
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        mrr[f"MRR"] = round(mrr[f"MRR"] / len(scores), 5)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision, mrr]:
            logging.info("\n")
            for k in eval.keys():
                huh = f"{k}: {eval[k]:.4f}"
                logging.info(f"{k}: {eval[k]:.4f}")

        return ndcg, _map, recall, precision, mrr
