from transformers import RobertaConfig, get_scheduler,PretrainedConfig
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset

import argparse
import wandb
import time
import json

from preprocessing import *
from models import *

def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config_path', default="/u/poellhul/Documents/Masters/benchmarkIR/src/configs/default.json")
    #argparser.add_argument('--input_dir', default="/part/01/Tmp/lvpoellhuber/datasets")
    #argparser.add_argument('--model_dir', default="/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm")
    #argparser.add_argument('--tokenizer_dir', default="/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm")
    #argparser.add_argument('--batch_size', default=16) # TODO: same
    #argparser.add_argument('--checkpoint', default=None)
    #argparser.add_argument('--attention', default="default") # default, adaptive, bpt, dynamic
    #argparser.add_argument('--logging', default=False) # default, adaptive, dtp

    args = argparser.parse_args()

    return args

def download_dataset(data_path):
    # Creating sample files to simplify tokenizer training. 
    if not os.path.exists(os.path.join(data_path, "wiki_en_0.txt")):
        dataset = load_dataset("wikipedia", language="en", date="20220301", cache_dir=data_path)
        create_sample_files(dataset, data_path)
    paths = [str(x) for x in Path(data_path).glob("*.txt")]

    return paths

def get_tokenizer(paths, tokenizer_path):
    # Tokenizer training
    if os.path.exists(os.path.join(tokenizer_path, "vocab.json")):
        print("Loading pretrained tokenizer.")
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    else:
        print("Training tokenizer.")
        tokenizer = train_BPETokenizer(paths, tokenizer_path)

    return tokenizer

def get_dataloader(tokenizer, paths, batch_size, dataset_path):
    dataset = generate_dataset(tokenizer, paths, dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)

    return dataloader


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    with open(args.config_path) as fp: arg_dict = json.load(fp)

    config = arg_dict["config"]
    settings = arg_dict["settings"]

    # Main arguments
    data_path = settings["datapath"]
    dataset_path = settings["dataset"]
    model_path = settings["model"]
    tokenizer_path = settings["tokenizer"]
    chkpt_path = os.path.join(model_path, "checkpoints")

    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device 

    # Generate or load dataset + tokenizer
    paths = download_dataset(data_path)
    tokenizer = get_tokenizer(paths, tokenizer_path)
    dataloader = get_dataloader(tokenizer, paths, settings["batch_size"], dataset_path)
    
    # Read the config
    print("Initializing training. ")
    config = RobertaConfig.from_dict(config)
    config.vocab_size = tokenizer.vocab_size+4
    print(config)

    model = RobertaForMaskedLM(config)
    model.to(device)

    optim = AdamW(model.parameters(), settings["lr"]) # typical range is 1e-6 to 1e-4

    # Number of training epochs and warmup steps
    epochs = 2
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

    print("Beginnig training process. ")
    # WandB stuff
    if settings["logging"]:
        wandb.login(key=os.getenv("WANDB_KEY"))
        run = wandb.init(
            project = "benchmarkIR",
            config = vars(settings + config)
        )
        accelerator.init_trackers("benchmarkIR")

    # You can add a config here, for the experiment
    
    if settings["use_checkpoint"]:
        accelerator.print(f"Resumed from checkpoint: {chkpt_path}")
        accelerator.load_state(chkpt_path, strict=False)

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
            optim.step()
            scheduler.step()
            #print("\n\n\n\nStep\n\n\n\n")
            #if i%1000==0:
            #wandb.log({"loss": loss})
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