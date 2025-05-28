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
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_scheduler

from preprocessing.preprocess_utils import get_pairs_dataloader, get_triplets_dataloader
from model_biencoder import LongBiEncoder
from modeling_utils import *
from losses import *

import datetime

import dotenv
dotenv.load_dotenv()
torch.autograd.set_detect_anomaly(True)
STORAGE_DIR = os.getenv("STORAGE_DIR")

def debug_accelerate(train_dataloader, biencoder):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    q_device = biencoder.q_model.device
    doc_device = biencoder.doc_model.device
    log_message(f"| Local rank: {local_rank} --- Question encoder on {q_device} --- Doc encoder on {doc_device}", logging.DEBUG)
    for batch_idx, batch in enumerate(train_dataloader):
        log_message(f"| GPU {torch.cuda.current_device()} has batch {batch_idx}", logging.DEBUG)
    log_message("_________________\n", logging.DEBUG)


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="longtriever_debug") 
    argparser.add_argument('--config_dict', default={})
    
    args = argparser.parse_args()

    return args


def train_longtriever(arg_dict):
    settings = arg_dict["settings"]
    config_dict = arg_dict["config"]
    

    # Main arguments
    shared_encoder = config_dict["shared_encoder"]
    if shared_encoder:
        model_path = config_dict["q_model"]
    else:
        q_model_path = config_dict["q_model"]
        doc_model_path = config_dict["doc_model"]

    gradient_clipping = False
    if ("doc_gradient_clipping" in config_dict) & ("q_gradient_clipping" in config_dict):
        log_message("Using gradient clipping. ")
        q_gradient_clipping = config_dict["q_gradient_clipping"]
        doc_gradient_clipping = config_dict["doc_gradient_clipping"]
        gradient_clipping = True

    save_path = settings["save_path"]
    use_negatives = settings["negatives"]
    batch_size = settings["batch_size"]
    task = settings["task"]
    exp_name = settings["exp_name"]
    enable_logging = settings["logging"]
    checkpoint_steps = settings["checkpoint_steps"]
    resume_from_checkpoint = "checkpoint" in config_dict
    gradient_accumulation_steps = settings["gradient_accumulation_steps"]
    logging_steps = settings["logging_steps"] 

    log_note_path = os.path.join(save_path, "slurm_ids.txt")
    with open(log_note_path, "a") as log_file:
        slurm_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else "local"
        log_file.write(f"Training Job Slurm ID: {slurm_id}; Computer: {os.uname()[1]}\n")

    def seed_everything(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    seed = settings["seed"]
    seed_everything(seed)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    log_message(f"========================= Finetuning run {exp_name}.=========================")

    # Read the config
    log_message("Initializing training. ")
    
    task_datapath = os.path.join(STORAGE_DIR, os.path.join("datasets", task))
    if use_negatives:
        dataset_path = os.path.join(task_datapath, "train_triplets.pt")
        dataloader = get_triplets_dataloader(
            batch_size=batch_size, 
            dataset_path=dataset_path, 
            pin_memory=True, 
            prefetch_factor=2, 
            num_workers = 4
        )
         # TODO: add to config dict
        # loss_fct = contrastive_loss
    else: 
        dataset_path = os.path.join(task_datapath, "train_pairs.pt")
        dataloader = get_pairs_dataloader(
            batch_size=batch_size, 
            dataset_path=dataset_path, 
            pin_memory=True, 
            prefetch_factor=2, 
            num_workers = 4
        )
    loss_fct = LOSS_FUNCTIONS[settings["loss_fn"]]

    biencoder = LongBiEncoder(
        model_path=(q_model_path, doc_model_path) if not shared_encoder else model_path,
        normalize=config_dict["normalize"],
        prompts={"query": config_dict["query_prompt"], "passage": config_dict["passage_prompt"]},
        attn_implementation=config_dict["attn_implementation"], 
        sep = config_dict["sep"], 
        batch_size=settings["batch_size"],
        max_block_length=config_dict["max_block_length"], 
        max_num_blocks=config_dict["max_num_blocks"],
        model_type = config_dict["model_type"] if "model_type" in config_dict else "longtriever"
    )
    biencoder.train()
    
    if not shared_encoder:
        optim = AdamW([
            {"params": biencoder.q_model.parameters(), 'lr': settings["lr"]},
            {"params": biencoder.doc_model.parameters(), 'lr': settings["lr"]} 
            ], 
        betas=(settings["adam_beta1"], settings["adam_beta2"]), 
        eps = settings["adam_epsilon"], 
        weight_decay=settings["weight_decay"]
        ) 
    else:
        optim = AdamW([
            {"params": biencoder.q_model.parameters(), 'lr': settings["lr"]},
        ]) 

    # Number of training epochs and warmup steps
    epochs = settings["epochs"]
    num_training_steps = (epochs * len(dataloader)) / gradient_accumulation_steps
    num_warmup_steps = int(settings["warmup_ratio"] * num_training_steps)

    # Initialize the scheduler
    scheduler = get_scheduler(
        settings["lr_scheduler_type"], 
        optimizer=optim, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )


    experiment = None
    if enable_logging & accelerator.is_main_process:
        experiment  = comet_ml.Experiment(project_name="new-attention", auto_metric_step_rate=logging_steps)
        exp_name = f"{settings['exp_name']}_{task}"
        experiment.set_name(exp_name)
        experiment.set_model_graph(biencoder.__dict__)

   

    optim, dataloader, scheduler = biencoder.accelerate_model(optim, dataloader, scheduler, accelerator)


    resume_step = None
    if resume_from_checkpoint:
        accelerator.load_state(config_dict["checkpoint"])
        resume_step = scheduler.scheduler.last_epoch
        starting_epoch = resume_step // len(dataloader)
        resume_step -= starting_epoch * len(dataloader)
    
    log_message("Beginning training process. ")
    # Training loop
    overall_step = 0 # Independent from i, because i restart at each epoch.
    for epoch in range(epochs):
        if resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step only if we are not using a stateful dataloader
            active_dataloader = accelerator.skip_first_batches(dataloader, resume_step)
            overall_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = dataloader
        loop = tqdm(active_dataloader, desc=f'Epoch {epoch} ({accelerator.process_index})')
        for i, batch in enumerate(loop):
            with accelerator.accumulate([biencoder.q_model, biencoder.doc_model]):
                # big_tic = time.time()
                overall_step+=1

                # Step init
                optim.zero_grad()

                # Input
                queries = batch["queries"]
                documents = batch["documents"]


                # Encoding
                # tic = time.time()
                q_embeddings = biencoder.encode_queries(queries, convert_to_tensor=True) 
                # toc = time.time()
                # log_message(f"Encoding batch queries: {toc-tic:.5f}s")
                # tic = time.time()
                doc_embeddings = biencoder.encode_corpus(documents, convert_to_tensor=True) 
                # toc = time.time()
                # log_message(f"Encoding batch corpus: {toc-tic:.5f}s")

                # Backwards pass
                # tic = time.time()
                loss = loss_fct(q_embeddings, doc_embeddings)
                # toc = time.time()
                # log_message(f"Loss function: {toc-tic:.5f}s")
                # tic = time.time()
                accelerator.backward(loss)
                # toc = time.time()
                # log_message(f"Backwards pass: {toc-tic:.5f}s")

                # scan_gradient_norm(biencoder, overall_step, batch, save_path)

                if gradient_clipping:
                    accelerator.clip_grad_norm_(biencoder.q_model.parameters(), max_norm=q_gradient_clipping)
                    accelerator.clip_grad_norm_(biencoder.doc_model.parameters(), max_norm=doc_gradient_clipping)


                # Updates
                optim.step()
                scheduler.step()
                
                # Logging info
                if (overall_step % logging_steps==0) & enable_logging & accelerator.is_main_process:
                    unwrapped_q_model = accelerator.unwrap_model(biencoder.q_model)
                    unwrapped_doc_model = accelerator.unwrap_model(biencoder.doc_model)
                    log_metrics(unwrapped_q_model, unwrapped_doc_model, scheduler, optim, experiment, loss, overall_step)

                # Saving checkpoint
                if overall_step%checkpoint_steps==0:
                    biencoder.save_checkpoint(save_path, accelerator)
                    accelerator.wait_for_everyone()

                # TQDM stuff
                now = datetime.datetime.now().strftime("%H:%M:%S")
                loop.set_postfix(loss = loss.item(), time=now, refresh=False)

                # big_toc = time.time()
                # log_message(f"===========================================================================\nStep {overall_step}: {big_toc-big_tic:.5f}s")

        # Free up memory 
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Save every epoch        
        biencoder.save_checkpoint(save_path, accelerator, checkpoint_dir_name="epoch_checkpoints")
        accelerator.wait_for_everyone()

    # Save at the end of training
    accelerator.wait_for_everyone()
    log_message("Training done. Saving model. ")
    biencoder.save_pretrained(
        save_path, 
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        accelerator=accelerator 
    )

    # Ending modules
    accelerator.free_memory()
    del biencoder, optim, dataloader 
    torch.cuda.empty_cache()   
    comet_ml.end() 

if __name__ == "__main__":    
    args = parse_arguments()

    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+".json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    arg_dict = default_args(arg_dict)

    train_longtriever(arg_dict)