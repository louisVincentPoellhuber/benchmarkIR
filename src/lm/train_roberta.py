from transformers import RobertaConfig, get_scheduler,PretrainedConfig
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs
from roberta_config import CustomRobertaConfig

import argparse
import wandb
from tqdm import tqdm
import json
import os

from preprocessing import preprocess_main, get_dataloader, get_tokenizer
from model_custom_roberta import *

def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="adaptive") # default, adaptive, sparse
    
    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR/src/lm/configs", args.config+"_mlm.json")
    with open(config_path) as fp: arg_dict = json.load(fp)

    config = arg_dict["config"]
    settings = arg_dict["settings"]
    train_args = arg_dict["train_args"]
    preprocess_args = arg_dict["preprocess_args"]
    enable_accelerate = settings["accelerate"]

    # Main arguments
    data_path = settings["datapath"]
    dataset_path = train_args["dataset"]
    model_path = settings["model"]
    tokenizer_path = settings["tokenizer"]
    chkpt_path = settings["checkpoint"] if settings["checkpoint"] != None else os.path.join(model_path, "checkpoints")

    if enable_accelerate:
        print("Accelerate enabled. ")
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        device = accelerator.device 
    else:
        print("Accelerate disabled.")
        device = "cuda" if torch.cuda.is_available() else "cpu"


    # Generate or load dataset + tokenizer
    if preprocess_args["preprocess"]:
        preprocess_main(preprocess_args["dataset"], data_path, tokenizer_path, preprocess_args["train_tokenizer"], preprocess_args["overwrite"])

    tokenizer = get_tokenizer(tokenizer_path)
    dataloader = get_dataloader(train_args["batch_size"], dataset_path)
    
    # Read the config
    print("Initializing training. ")
    config = CustomRobertaConfig.from_dict(config)
    config.vocab_size = tokenizer.vocab_size+4 
    print(config)

    model = RobertaForMaskedLM(config)
    model.to(device)


    # For sparse attention 
    if config.attn_mechanism == "sparse":
        alpha_params = []
        for i in range(len(model.roberta.encoder.layer)):
            alpha_params.append(model.roberta.encoder.layer[i].attention.self.alpha)

        other_params = []
        for name, param in model.named_parameters():
            if not name.endswith('attention.self.alpha'):
                other_params.append(param)
        optim = AdamW([
                {'params': other_params, 'lr': 1e-5},  # Default learning rate for most parameters
                {'params': alpha_params, 'lr': 10.0}          # Higher learning rate for alpha
            ])
    else:
        optim = AdamW(model.parameters(), lr=train_args["lr"]) # typical range is 1e-6 to 1e-4

    # Number of training epochs and warmup steps
    epochs = train_args["epochs"]
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    # Initialize the scheduler
    scheduler = get_scheduler(
        "linear", 
        optimizer=optim, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    if enable_accelerate:
        # Accelerator function  
        model, optim, dataloader, scheduler = accelerator.prepare(
            model, optim, dataloader, scheduler
        )

    print("Beginning training process. ")
    # WandB stuff
    if train_args["logging"]:
        wandb.login(key=os.getenv("WANDB_KEY"))
        run = wandb.init(
            project = "benchmarkIR",
            config = vars(config)
        )
        
        if enable_accelerate: accelerator.init_trackers("benchmarkIR")

    # You can add a config here, for the experiment
   
    #if train_args["use_checkpoint"]: # TODO: Figure out how to use checkpoint
    #    accelerator.print(f"Resumed from checkpoint: {chkpt_path}")
    #    accelerator.load_state(chkpt_path, strict=False)

    # Training loop
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            
            if enable_accelerate:
                input_ids = batch["input_ids"]
                mask = batch["attention_mask"]
                labels = batch["labels"]
            else:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask = mask, labels = labels) # All three 16x512

            loss = outputs.loss

            if enable_accelerate:
                accelerator.backward(loss)
            else:
                loss.backward() 

            #print(model.roberta.encoder.layer[0].attention.self.seq_attention.mask.current_val)

            optim.step()
            
            scheduler.step()
            if enable_accelerate:
                if (i%10000==0) & (i!=0):
                    accelerator.save_state(chkpt_path)
                
                if config.attn_mechanism =="sparse":
                    original_model = accelerator.unwrap_model(model)
                    log_dict = {"loss": loss}
                    for layer_nb, layer in enumerate(original_model.roberta.encoder.layer):
                        alphas = layer.attention.self.alpha.data
                        names = [f"layer_{layer_nb}_alpha_"+str(i) for i in range(len(alphas))]
                        alpha_dict = dict(zip(names, alphas))
                        log_dict.update(alpha_dict)
                    accelerator.log(log_dict)

                elif config.attn_mechanism == "adaptive":
                    original_model = accelerator.unwrap_model(model)
                    log_dict = {"loss": loss}
                    for layer_nb, layer in enumerate(original_model.roberta.encoder.layer):
                        spans = layer.attention.self.seq_attention.mask.current_val.data
                        names = [f"layer_{layer_nb}_span_"+str(i) for i in range(len(spans))]
                        span_dict = dict(zip(names, spans))
                        log_dict.update(span_dict)
                    accelerator.log(log_dict)

                else:
                    accelerator.log({"loss": loss})

            elif train_args["logging"]:
                wandb.log({"loss": loss})

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss = loss.item())

    print("Training done. Saving model. ")
    if enable_accelerate:
        accelerator.save_state(chkpt_path)
        accelerator.end_training()
    
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            model_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
    else:
        model.save_pretrained(model_path, from_pt = True)
