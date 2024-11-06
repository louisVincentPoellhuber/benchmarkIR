import comet_ml

from preprocessing import *
from model_custom_roberta import *

import argparse
import pandas as pd
import json

import wandb
from transformers import RobertaConfig, get_scheduler
from roberta_config import CustomRobertaConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from datasets import load_dataset, load_metric
import copy

def parse_arguments():
    argparser = argparse.ArgumentParser("Evaluating Roberta")
    argparser.add_argument('--config', default="roberta_glue") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) 
    args = argparser.parse_args()

    return args

def log_metrics(accelerator, model, experiment, step, epoch, num_training_steps, config, loss):
    step = step + epoch * num_training_steps
    if experiment != None:
        experiment.log_metrics({"loss": loss}, step=step)
        
        if accelerator != None:                
            if accelerator.is_main_process:
                if config.attn_mechanism =="sparse":
                    original_model = accelerator.unwrap_model(model)
                    log_dict = {"loss": loss}
                    for layer_nb, layer in enumerate(original_model.roberta.encoder.layer):
                        alphas = layer.attention.self.alpha.data
                        names = [f"layer_{layer_nb}_alpha_"+str(i) for i in range(len(alphas))]
                        alpha_dict = dict(zip(names, alphas))
                        log_dict.update(alpha_dict)
                    experiment.log_metrics(log_dict, step=step)

                elif config.attn_mechanism == "adaptive":
                    original_model = accelerator.unwrap_model(model)
                    log_dict = {"loss": loss}
                    for layer_nb, layer in enumerate(original_model.roberta.encoder.layer):
                        spans = layer.attention.self.seq_attention.mask.current_val.data
                        names = [f"layer_{layer_nb}_span_"+str(i) for i in range(len(spans))]
                        span_dict = dict(zip(names, spans))
                        log_dict.update(span_dict)
                    experiment.log_metrics(log_dict, step=step)
        else:
                if config.attn_mechanism =="sparse":
                    log_dict = {"loss": loss}
                    for layer_nb, layer in enumerate(model.roberta.encoder.layer):
                        alphas = layer.attention.self.alpha.data
                        names = [f"layer_{layer_nb}_alpha_"+str(i) for i in range(len(alphas))]
                        alpha_dict = dict(zip(names, alphas))
                        log_dict.update(alpha_dict)
                    experiment.log_metrics(log_dict, step=step)

                elif config.attn_mechanism == "adaptive":
                    log_dict = {"loss": loss}
                    for layer_nb, layer in enumerate(model.roberta.encoder.layer):
                        spans = layer.attention.self.seq_attention.mask.current_val.data
                        names = [f"layer_{layer_nb}_span_"+str(i) for i in range(len(spans))]
                        span_dict = dict(zip(names, spans))
                        log_dict.update(span_dict)
                    experiment.log_metrics(log_dict, step=step)

def main(arg_dict):
    config_path = arg_dict["config"]
    settings = arg_dict["settings"]
    train_args = arg_dict["train_args"]
    eval_args = arg_dict["eval_args"]
    enable_accelerate = settings["accelerate"]
    preprocess_args = arg_dict["preprocess_args"]

    # Main arguments
    data_path = settings["datapath"]
    dataset_path = train_args["dataset"]
    model_path = settings["model"]
    model_save_path = settings["save_path"]
    tokenizer_path = settings["tokenizer"]
    chkpt_path = settings["checkpoint"] if not None else os.path.join(model_path, "checkpoints")

    task = preprocess_args["task"]

    accelerator = Accelerator(log_with="comet_ml", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device 
    
    task_num_labels = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "qnli": 2,
        "qqp": 2,
        "rte": 2, 
        "sst2": 2,
        "wnli": 2, 
        "stsb": 1
    }
    # Generate or load dataset + tokenizer
    if preprocess_args["preprocess"]:
        preprocess_main(preprocess_args["dataset"], data_path, tokenizer_path, preprocess_args["train_tokenizer"], preprocess_args["overwrite"])

    metric = load_metric("glue", task, trust_remote_code=True)
    accuracy = load_metric("accuracy", trust_remote_code = True)
    tokenizer = get_tokenizer(tokenizer_path)
    
    if train_args["train"]:   

        
        print("Initializing training. ")
        config = CustomRobertaConfig.from_dict(config_path)
        config.vocab_size = tokenizer.vocab_size
        config.num_labels = task_num_labels[task]

        model = RobertaForSequenceClassification(config=config).from_pretrained(chkpt_path, config=config, ignore_mismatched_sizes=True)
        model.to(device)
        print(model.config.num_labels)

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
                    {'params': other_params, 'lr': train_args["lr"]},  # Default learning rate for most parameters
                    {'params': alpha_params, 'lr': train_args["alpha_lr"]}          # Higher learning rate for alpha
                ], betas=(0.9, 0.98), eps=1e-6)
        elif config.attn_mechanism == "adaptive":
            adaptive_params = []
            for i in range(len(model.roberta.encoder.layer)):
                adaptive_params.append(model.roberta.encoder.layer[i].attention.self.alpha)

            other_params = []
            for name, param in model.named_parameters():
                if not name.endswith('attention.self.seq_attention.adaptive_span._mask.current_val'):
                    other_params.append(param)
            optim = AdamW([
                    {'params': other_params, 'lr': train_args["lr"]},  # Default learning rate for most parameters
                    {'params': adaptive_params, 'lr': train_args["adaptive_lr"]}          # Higher learning rate for alpha
                ], betas=(0.9, 0.98), eps=1e-6)
        else:
            optim = AdamW(model.parameters(), lr=train_args["lr"]) # typical range is 1e-6 to 1e-4
        
        train_dataloader = get_dataloader(train_args["batch_size"], dataset_path)

        # Number of training epochs and warmup steps
        epochs = train_args["epochs"]
        num_training_steps = epochs * len(train_dataloader)
        num_warmup_steps = int(0.06 * num_training_steps)

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
            experiment = None
            if enable_accelerate: 
                accelerator.init_trackers(project_name="benchmarkIR", config = config.to_dict())
                if accelerator.is_main_process:
                    experiment  = comet_ml.Experiment(project_name="benchmarkIR", auto_metric_step_rate=100)
                    experiment.set_name(f"{train_args["exp_name"]}_{task}")
            else:
                experiment = comet_ml.Experiment(project_name="benchmarkIR", auto_metric_step_rate=100)
                experiment.set_name(f"{train_args["exp_name"]}_{task}")



        # Training loop
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True)
            for i, batch in enumerate(loop):
                optim.zero_grad()
                input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
                mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
                labels = batch["labels"]#.to(device)

                outputs = model(input_ids, attention_mask = mask, labels = labels)

                loss = outputs.loss
                #loss.backward() # again, replaced by the accelerator version
                accelerator.backward(loss)
                optim.step()
                scheduler.step()


                loop.set_description(f'Epoch: {epoch}')
                loop.set_postfix(loss = loss.item())

                
                if (i%100==0) & train_args["logging"]:
                    log_metrics(accelerator, model, experiment, i, epoch, len(loop), config, loss)
                
            #unwrapped_model = accelerator.unwrap_model(model)
            #unwrapped_model.save_pretrained(
            #    model_save_path, # used to be model_path
            #    is_main_process=accelerator.is_main_process,
            #    save_function=accelerator.save,
            #)

        print("Training done. Saving model. ")
        accelerator.end_training()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            model_save_path, # used to be model_path
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    if eval_args["eval"]:    
        test_dataloader = get_dataloader(train_args["batch_size"], eval_args["dataset"])

        config = CustomRobertaConfig.from_dict(config_path)
        config.vocab_size = tokenizer.vocab_size
        config.num_labels = task_num_labels[task]
        print(config.num_labels)

        model = RobertaForSequenceClassification(config=config).from_pretrained(eval_args["model"], config=config, ignore_mismatched_sizes=True)
        model.to(device)

        # Accelerator function
        model, dataloader = accelerator.prepare(
            model, test_dataloader
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

            row = [float(outputs.loss)] + list(metrics.values())
            row_names = ["loss"] + list(metrics.keys())

            if "accuracy" not in row_names:
                train_acc = accuracy.compute(predictions=predictions, references=labels)
                row.append(train_acc["accuracy"])
                row_names.append("accuracy")
            metrics_df.append(row)
        
        metrics_df = pd.DataFrame(metrics_df, columns = row_names)
        metrics_df.to_csv(os.path.join(model_save_path, "metrics.csv"))

if __name__ == "__main__":
    args = parse_arguments()
    
    if len(args.config_dict)>0:
        original_arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR/src/lm/configs", args.config+"_finetune.json")
        with open(config_path) as fp: 
            original_arg_dict = json.load(fp)

    if original_arg_dict["preprocess_args"]["task"]=="glue":
        tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]
        

        for task in tasks:
            print(f"============ Processing {task} ============")

            # Adjusting the config for each task
            arg_dict = copy.deepcopy(original_arg_dict)

            if arg_dict["eval_args"]["model"] != "FacebookAI/roberta-base":
                arg_dict["eval_args"]["model"] = os.path.join(arg_dict["eval_args"]["model"], "roberta_"+task)
            else:
                for task in tasks:
                    save_path = os.path.join(arg_dict["settings"]["save_path"], "roberta_"+task)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

            arg_dict["preprocess_args"]["task"] = task
            
            arg_dict["settings"]["datapath"] = os.path.join(arg_dict["settings"]["datapath"], task)
            arg_dict["settings"]["save_path"] = os.path.join(arg_dict["settings"]["save_path"], "roberta_"+task)  # Changed 'save_path' from 'model'
            
            arg_dict["train_args"]["dataset"] = os.path.join(arg_dict["settings"]["datapath"], task+"_train.pt")
            
            arg_dict["eval_args"]["dataset"] = os.path.join(arg_dict["settings"]["datapath"], task+"_test.pt")

            main(arg_dict)
        
    else:
        main(original_arg_dict)