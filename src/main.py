from transformers import RobertaConfig, get_scheduler,PretrainedConfig
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs

import argparse
import wandb
from tqdm import tqdm
import json
import os

from preprocessing import preprocess_main, get_dataloader, get_tokenizer
from models import *

def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="sparse") # default, adaptive, dtp
    
    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR/src/configs", args.config+"_mlm.json")
    with open(config_path) as fp: arg_dict = json.load(fp)

    config = arg_dict["config"]
    settings = arg_dict["settings"]
    train_args = arg_dict["train_args"]
    preprocess_args = arg_dict["preprocess_args"]

    # Main arguments
    data_path = settings["datapath"]
    dataset_path = train_args["dataset"]
    model_path = settings["model"]
    tokenizer_path = settings["tokenizer"]
    chkpt_path = settings["checkpoint"] if not None else os.path.join(model_path, "checkpoints")

    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device 


    # Generate or load dataset + tokenizer
    if preprocess_args["preprocess"]:
        preprocess_main(preprocess_args["dataset"], data_path, tokenizer_path, preprocess_args["train_tokenizer"], preprocess_args["overwrite"])

    tokenizer = get_tokenizer(tokenizer_path)
    dataloader = get_dataloader(train_args["batch_size"], dataset_path)
    
    # Read the config
    print("Initializing training. ")
    config = RobertaConfig.from_dict(config)
    config.vocab_size = tokenizer.vocab_size+4
    print(config)

    model = RobertaForMaskedLM(config)
    model.to(device)

    optim = AdamW(model.parameters(), train_args["lr"]) # typical range is 1e-6 to 1e-4

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
            config = vars(settings + config)
        )
        accelerator.init_trackers("benchmarkIR")

    # You can add a config here, for the experiment
    
    #if train_args["use_checkpoint"]: # TODO: Figure out how to use checkpoint
    #    accelerator.print(f"Resumed from checkpoint: {chkpt_path}")
    #    accelerator.load_state(chkpt_path, strict=False)

    # Training loop
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for i, batch in enumerate(loop):
            print(loop)
            optim.zero_grad()
            input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
            mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
            labels = batch["labels"]#.to(device)

            #print("\nTraining loop. ")
            #print(f"Input IDs: {input_ids.shape}")
            #print(f"Mask: {mask.shape}")
            #print(f"Labels: {labels.shape}")
            outputs = model(input_ids, attention_mask = mask, labels = labels) # All three 16x512

            loss = outputs.loss
            #loss.backward() # again, replaced by the accelerator version
            accelerator.backward(loss)
            print(model.roberta.encoder.layer[0].attention.self.alpha.grad)
            #print(model.roberta.encoder.layer[0].attention.self.seq_attention.adaptive_span._mask.current_val.grad)
            optim.step()
            scheduler.step()

            accelerator.log({"loss": loss})
            if (i%10000==0) & (i!=0):
                accelerator.save_state(chkpt_path)

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss = loss.item())

    accelerator.save_state(chkpt_path)
    accelerator.end_training()
    
    print("Training done. Saving model. ")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        model_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
