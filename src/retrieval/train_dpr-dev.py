import comet_ml

from preprocessing import get_pairs_dataloader, get_triplets_dataloader, get_tokenizer
from losses import contrastive_loss
from model_custom_dpr import CustomDPR, CustomRobertaConfig, RobertaForSequenceClassification, CustomRobertaModel
from dpr_config import CustomDPRConfig
from beir.retrieval.models import SentenceBERT
from accelerate import Accelerator, DistributedDataParallelKwargs
from model_biencoder import BiEncoder

import torch
from torch.optim import AdamW
from transformers import get_scheduler

import os
import json
from tqdm import tqdm
import argparse

from modeling_utils import *

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="default") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) # default, adaptive, sparse
    
    args = argparser.parse_args()

    return args

def train_dpr(arg_dict):
    config = arg_dict["config"]
    settings = arg_dict["settings"]
    
    enable_accelerate = settings["accelerate"]
    logging = settings["logging"]

    # Main arguments
    dataset_path = settings["dataset"]
    q_model_path = settings["q_model"]
    doc_model_path = settings["doc_model"]
    model_path = settings["save_path"]
    tokenizer_path = settings["tokenizer"]

    task = settings["task"]

    accelerator = Accelerator(log_with="comet_ml", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    dataloader = get_triplets_dataloader(batch_size=16, dataset_path=dataset_path)

    # Read the config
    print("Initializing training. ")
    # config = CustomDPRConfig.from_dict(config)
    # config.vocab_size = tokenizer.vocab_size
    # print(config)

    # dpr_model = SentenceBERT(model_path = (q_model_path, doc_model_path), sep=" [SEP] ")
    dpr_model = BiEncoder(model_path = (q_model_path, doc_model_path), sep=" [SEP] ")
    dpr_model.train()
    
    optim = AdamW([
        {"params": dpr_model.q_model.parameters(), 'lr': settings["lr"]},
        {"params": dpr_model.doc_model.parameters(), 'lr': settings["lr"]}
    ]) 

    # Number of training epochs and warmup steps
    epochs = settings["epochs"]
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    # Initialize the scheduler
    scheduler = get_scheduler(
        "linear", 
        optimizer=optim, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    dpr_model, optim, dataloader, scheduler = accelerator.prepare(
        dpr_model, optim, dataloader, scheduler
    )

    # TODO: add to config dict
    loss_fct = contrastive_loss

    # TODO: ensure comet logs things properly
    # experiment = comet_ml.Experiment(project_name="test", auto_metric_step_rate=100)


    print("Beginning training process. ")
    # Training loop
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            
            queries = batch["queries"]
            positives = batch["positive"]
            negatives = batch["negatives"]
            documents = positives + negatives

            q_embeddings = dpr_model.encode_queries(queries, convert_to_tensor=True) # All three 16x512
            doc_embeddings = dpr_model.encode_corpus(documents, convert_to_tensor=True) # All three 16x512

            loss = loss_fct(q_embeddings, doc_embeddings)
            #loss.backward()
            accelerator.backward(loss)

            optim.step()
            scheduler.step()

            # if (i%100==0) & logging:
            #     log_metrics(dpr_model, experiment, i, epoch)

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss = loss.item())

    print("Training done. Saving model. ")
   
    dpr_model.save_pretrained(model_path) 

if __name__ == "__main__":    
    args = parse_arguments()

    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+"_train.json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    
    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)

    train_dpr(arg_dict)