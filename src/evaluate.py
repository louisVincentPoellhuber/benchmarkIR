from preprocessing import *
from models import *

import argparse
import pandas as pd
import json

import wandb
from transformers import RobertaConfig, get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from datasets import load_dataset, load_metric

def parse_arguments():
    argparser = argparse.ArgumentParser("masked language modeling")
    argparser.add_argument('--config', default="adaptive") # default, adaptive, dtp
    args = argparser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR/src/configs", args.config+"_finetune.json")
    with open(config_path) as fp: arg_dict = json.load(fp)

    config = arg_dict["config"]
    settings = arg_dict["settings"]
    train_args = arg_dict["train_args"]
    eval_args = arg_dict["eval_args"]
    preprocess_args = arg_dict["preprocess_args"]

    # Main arguments
    data_path = settings["datapath"]
    dataset_path = train_args["dataset"]
    model_path = settings["model"]
    tokenizer_path = settings["tokenizer"]
    chkpt_path = settings["checkpoint"] if not None else os.path.join(model_path, "checkpoints")

    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device 
    
    # Generate or load dataset + tokenizer
    if preprocess_args["preprocess"]:
        preprocess_main(preprocess_args["dataset"], data_path, tokenizer_path, preprocess_args["train_tokenizer"], preprocess_args["overwrite"])

    metric = load_metric("glue", "mnli", trust_remote_code=True)
    tokenizer = get_tokenizer(tokenizer_path)
    
    if train_args["train"]:   
        train_dataloader = get_dataloader(train_args["batch_size"], dataset_path)

        
        print("Initializing training. ")
        config = RobertaConfig.from_dict(config)
        config.vocab_size = tokenizer.vocab_size+4


        model = RobertaForSequenceClassification(config=config).from_pretrained(chkpt_path, config=config)
        model.to(device)

        optim = AdamW(model.parameters(), lr=train_args["lr"]) # typical range is 1e-6 to 1e-4

        # Number of training epochs and warmup steps
        epochs = train_args["epochs"]
        num_training_steps = epochs * len(train_dataloader)
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
            model, optim, train_dataloader, scheduler
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

        # Training loop
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True)
            for i, batch in enumerate(loop):
                print(loop)
                optim.zero_grad()
                input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
                mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
                labels =batch["labels"]#.to(device)
                outputs = model(input_ids, attention_mask = mask, labels = labels)

                loss = outputs.loss
                #loss.backward() # again, replaced by the accelerator version
                accelerator.backward(loss)
                optim.step()
                scheduler.step()

                #if i%1000==0:
                #wandb.log({"loss": loss})
                accelerator.log({"loss": loss})
                if i%10000==0:
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

    if args["do_eval"]:    
        val_dataloader = get_dataloader(train_args["batch_size"], eval_args["dataset"])

        config = RobertaConfig.from_dict(config)
        config.vocab_size = tokenizer.vocab_size+4


        model = RobertaForSequenceClassification(config=config).from_pretrained(eval_args["model"], config=config)
        model.to(device)

        # Accelerator function
        model, dataloader = accelerator.prepare(
            model, val_dataloader
        )


        loop = tqdm(dataloader, leave=True)
        metrics_df = []
        for i, batch in enumerate(loop): 
            input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
            mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
            labels =batch["labels"]#.to(device)

            outputs = model(input_ids, attention_mask = mask, labels = labels)
            predictions = torch.argmax(outputs.logits, axis=1)
            metrics = metric.compute(predictions=predictions, references=labels)
       
            metrics_df.append([float(outputs.loss), metrics["accuracy"]])
        
        metrics_df = pd.DataFrame(metrics_df, columns = ["loss", "accuracy"])
        metrics_df.to_csv(os.path.join(model_path, "metrics.csv"))
        