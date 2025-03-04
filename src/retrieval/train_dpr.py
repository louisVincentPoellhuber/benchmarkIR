import comet_ml

from preprocessing import get_dataloader, get_tokenizer
from losses import contrastive_loss
from model_custom_dpr import CustomDPR, CustomRobertaConfig, RobertaForSequenceClassification, CustomRobertaModel
from dpr_config import CustomDPRConfig

import torch
from torch.optim import AdamW
from transformers import get_scheduler
from transformers.models.dpr import DPRQuestionEncoder

import os
import json
from tqdm import tqdm
import argparse


import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)


def debug_parameters(model, pretrained_path):
    pretrained_roberta = CustomRobertaModel.from_pretrained(pretrained_path).to(device)
    #pretrained_roberta.roberta.save_pretrained("/part/01/Tmp/lvpoellhuber/models/custom_roberta/test", from_pt = True) 
    # Compare weights
    #print(pretrained_roberta.state_dict())
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n Comparing weights \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    roberta_weights = pretrained_roberta.state_dict().keys()
   
    for name, param in model.q_model.named_parameters():
        shortened_name = name.replace("question_encoder.roberta_model.", "")
        if shortened_name in roberta_weights:
            roberta_param = pretrained_roberta.state_dict()[shortened_name]
            if torch.equal(param, roberta_param):
                #print(f"Layer {shortened_name} matches")
                pass
            else:
                print(f"Layer {shortened_name} does not match")
   
        else:
            print(f"Layer {name} not found in RoBERTa pre-trained weights.")


def debug_dataloader(dataloader):
    for batch in tqdm(dataloader, leave=True):
        for doc in batch["docs"]:
            text = doc["text"]
            if text.count(" ") + 1 >= 400:
                print("Need to truncate")

                
def log_metrics(model, experiment, step, epoch):
    experiment.log_metrics({"loss": loss}, step=step, epoch=epoch)
    
    if config.attn_mechanism =="sparse":
        log_dict = {"loss": loss}
        for layer_nb, layer in enumerate(model.roberta.encoder.layer):
            alphas = layer.attention.self.alpha.data
            names = [f"layer_{layer_nb}_alpha_"+str(i) for i in range(len(alphas))]
            alpha_dict = dict(zip(names, alphas))
            log_dict.update(alpha_dict)
        experiment.log_metrics(log_dict, step=step, epoch=epoch)

    elif config.attn_mechanism == "adaptive":
        log_dict = {"loss": loss}
        for layer_nb, layer in enumerate(model.roberta.encoder.layer):
            spans = layer.attention.self.seq_attention.mask.current_val.data
            names = [f"layer_{layer_nb}_span_"+str(i) for i in range(len(spans))]
            span_dict = dict(zip(names, spans))
            log_dict.update(span_dict)
        experiment.log_metrics(log_dict, step=step, epoch=epoch)


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="default") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) # default, adaptive, sparse
    
    args = argparser.parse_args()

    return args


if __name__ == "__main__":    
    args = parse_arguments()
    
    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+"_paired.json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    
    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)


    config = arg_dict["config"]
    settings = arg_dict["settings"]
    
    enable_accelerate = settings["accelerate"]
    logging = settings["logging"]

    # Main arguments
    dataset_path = settings["dataset"]
    q_model_path = settings["q_model"]
    ctx_model_path = settings["ctx_model"]
    model_path = settings["save_path"]
    tokenizer_path = settings["tokenizer"]

    task = settings["task"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    #device="cpu"  


    dataloader = get_dataloader(batch_size=16, dataset_path=dataset_path)
    tokenizer = get_tokenizer(tokenizer_path)
    
    # Read the config
    print("Initializing training. ")
    config = CustomDPRConfig.from_dict(config)
    config.vocab_size = tokenizer.vocab_size
    print(config)

    #model = DPR(("facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-ctx_encoder-single-nq-base"))
    model = CustomDPR(config=config, model_path=(q_model_path, ctx_model_path), device=device)
    
    #debug_parameters(model, q_model_path)
    #debug_dataloader(dataloader)

    optim = AdamW([
        {"params": model.q_model.parameters(), 'lr': 1e-5},
        {"params": model.ctx_model.parameters(), 'lr': 1e-5}
    ]) # typical range is 1e-6 to 1e-4

    # Number of training epochs and warmup steps
    epochs = 5
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    # Initialize the scheduler
    scheduler = get_scheduler(
        "linear", 
        optimizer=optim, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    loss_fct = contrastive_loss

    experiment = comet_ml.Experiment(project_name="implement-sparse", auto_metric_step_rate=100)


    print("Beginning training process. ")
    # Training loop
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            
            queries = batch["queries"]
            docs = batch["documents"]

            q_embeddings = model.encode_queries(queries) # All three 16x512
            ctx_embeddings = model.encode_corpus(docs) # All three 16x512

            loss = loss_fct(q_embeddings, ctx_embeddings)
            loss.backward()
            #print(model.roberta.encoder.layer[0].attention.self.seq_attention.mask.current_val)

            optim.step()
            
            scheduler.step()

            if (i%100==0) & logging:
                log_metrics(model, experiment, i, epoch)

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss = loss.item())

    print("Training done. Saving model. ")
   
    model.save_pretrained(model_path) 
