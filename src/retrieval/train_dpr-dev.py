import comet_ml

import argparse
import random
import gc
import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from transformers import get_scheduler
from beir.retrieval.models import SentenceBERT

from preprocessing import get_pairs_dataloader, get_triplets_dataloader, get_tokenized_dataloader, get_pairs_dataset, custom_pairs_collate_fn
from model_biencoder import BiEncoder
from modeling_utils import *
from losses import *

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

def debug_accelerate(train_dataloader, dpr_model):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    q_device = dpr_model.q_model.device
    doc_device = dpr_model.doc_model.device
    log_message(f"| Local rank: {local_rank} --- Question encoder on {q_device} --- Doc encoder on {doc_device}", logging.DEBUG)
    for batch_idx, batch in enumerate(train_dataloader):
        log_message(f"| GPU {torch.cuda.current_device()} has batch {batch_idx}", logging.DEBUG)
    log_message("_________________\n", logging.DEBUG)


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="test") 
    argparser.add_argument('--config_dict', default={})
    
    args = argparser.parse_args()

    return args

def train_dpr(arg_dict):
    settings = arg_dict["settings"]
    config_dict = arg_dict["config"]
    
    enable_accelerate = settings["accelerate"]
    enable_logging = settings["logging"]

    # Main arguments
    q_model_path = config_dict["q_model"]
    doc_model_path = config_dict["doc_model"]
    sep = config_dict["sep"]

    model_path = settings["save_path"]
    use_negatives = settings["negatives"]
    batch_size = settings["batch_size"]

    task = settings["task"]
    exp_name = settings["exp_name"]

    log_message(f"========================= Finetuning run {exp_name}.=========================")

    
    def seed_everything(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    seed_everything(42)

    logging_steps = 10 
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    task_datapath = os.path.join(STORAGE_DIR, os.path.join("datasets", task))
    if use_negatives:
        dataset_path = os.path.join(task_datapath, "train_triplets.pt")
        dataloader = get_triplets_dataloader(batch_size=batch_size, dataset_path=dataset_path)
    else: 
        dataset_path = os.path.join(task_datapath, "train_pairs.pt")
        dataloader = get_pairs_dataloader(batch_size=batch_size, dataset_path=dataset_path)
    if False: # temporary if: this is for the dynamic tokens
        dataset_path = os.path.join(task_datapath, "train_pairs_dynamic.pt")
        dataloader = get_pairs_dataloader(batch_size=batch_size, dataset_path=dataset_path, tokenizer_path=q_model_path, sep=sep)


    # Read the config
    log_message("Initializing training. ")

    dpr_model = BiEncoder(
        model_path=(q_model_path, doc_model_path),
        normalize=config_dict["normalize"],
        prompts={"query": config_dict["query_prompt"], "passage": config_dict["passage_prompt"]},
        attn_implementation=config_dict["attn_implementation"], 
        sep = config_dict["sep"], 
        batch_size=settings["batch_size"]
    )
    dpr_model.train()
    
    optim = AdamW([
        {"params": dpr_model.q_model.parameters(), 'lr': settings["lr"]},
        {"params": dpr_model.doc_model.parameters(), 'lr': settings["lr"]}
    ]) 

    # Number of training epochs and warmup steps
    epochs = settings["epochs"]
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    log_message(f"Pre-accelerate dataloader length: {len(dataloader)}", logging.DEBUG)

    # Initialize the scheduler
    scheduler = get_scheduler(
        "linear", 
        optimizer=optim, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    experiment = None
    if enable_logging & accelerator.is_main_process:
        experiment  = comet_ml.Experiment(project_name="new-attention", auto_metric_step_rate=logging_steps)
        exp_name = f"{settings['exp_name']}_{task}"
        experiment.set_name(exp_name)
        experiment.set_model_graph(dpr_model.__dict__)

    optim, dataloader, scheduler = dpr_model.accelerate_model(optim, dataloader, scheduler, accelerator)

    # TODO: add to config dict
    # loss_fct = contrastive_loss
    loss_fct = InBatchNegativeLoss()

    # debug_accelerate(dataloader, dpr_model)

    log_message("Beginning training process. ")
    # Training loop
    step = 0
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            step+=1

            # Dynamic pairs code
            # query_input = batch["query_inputs"]
            # doc_input = batch["doc_inputs"]
            # q_embeddings = dpr_model.encode_tokenized_queries(query_input, convert_to_tensor=True)
            # doc_embeddings = dpr_model.encode_tokenized_corpus(doc_input, convert_to_tensor=True) 

            # Regular pairs code
            queries = batch["queries"]
            if use_negatives:
                positives = batch["positives"]
                negatives = batch["negatives"]
                documents = positives + negatives
            else: 
                documents = batch["documents"]

            q_embeddings = dpr_model.encode_queries(queries, convert_to_tensor=True) 
            doc_embeddings = dpr_model.encode_corpus(documents, convert_to_tensor=True) 


            loss = loss_fct(q_embeddings, doc_embeddings)
            #loss.backward()
            accelerator.backward(loss)

            optim.step()
            scheduler.step()
            
            # Logging info
            if (step % logging_steps==0) & enable_logging & accelerator.is_main_process:
                # log_message(f"Process {accelerator.process_index} reached wait_for_everyone()", flush=True)
                # accelerator.wait_for_everyone()
                # log_message(f"Process {accelerator.process_index} passed wait_for_everyone()", flush=True)
                unwrapped_q_model = accelerator.unwrap_model(dpr_model.q_model)
                unwrapped_doc_model = accelerator.unwrap_model(dpr_model.doc_model)
                log_metrics(unwrapped_q_model, unwrapped_doc_model, scheduler, optim, experiment, loss, step)

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss = loss.item())

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        dpr_model.save_pretrained(
            model_path, 
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            accelerator=accelerator 
        )

    log_message("Training done. Saving model. ")
    dpr_model.save_pretrained(
        model_path, 
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        accelerator=accelerator 
    )

    # Ending modules
    accelerator.free_memory()
    del dpr_model, optim, dataloader 
    torch.cuda.empty_cache()   
    comet_ml.end() 

if __name__ == "__main__":    
    args = parse_arguments()

    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+".json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    
    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)

    train_dpr(arg_dict)