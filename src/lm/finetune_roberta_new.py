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

from adagrad_with_grad_clip import AdagradWithGradClip

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)

def parse_arguments():
    argparser = argparse.ArgumentParser("Evaluating Roberta")
    argparser.add_argument('--config', default="test") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) 
    args = argparser.parse_args()

    return args

def log_metrics(accelerator, model, experiment, step, epoch, num_training_steps, config, loss, print_metrics=False):
    step = step + epoch * num_training_steps

    if accelerator != None:                
        if accelerator.is_main_process:
            model = accelerator.unwrap_model(model)
        else:
            experiment = None
                
    if experiment != None:
        experiment.log_metrics({"loss": loss}, step=step)
        
        if config.attn_mechanism =="sparse":
            log_dict = {"loss": loss}
            for layer_nb, layer in enumerate(model.roberta.encoder.layer):
                alphas = layer.attention.self.true_alpha.data
                names = [f"layer_{layer_nb}/alpha_"+str(i) for i in range(len(alphas))]
                alpha_dict = dict(zip(names, alphas))
                if print_metrics:  print(f"Layer {layer_nb}: {alpha_dict}")

                log_dict.update(alpha_dict)
            experiment.log_metrics(log_dict, step=step)

        elif config.attn_mechanism == "adaptive":
            log_dict = {"loss": loss}
            for layer_nb, layer in enumerate(model.roberta.encoder.layer):
                spans = layer.attention.self.adaptive_mask._mask.attn_span.data
                names = [f"layer_{layer_nb}/span_"+str(i) for i in range(len(spans))]
                span_dict = dict(zip(names, spans))
                log_dict.update(span_dict)

                if print_metrics:  print(f"Layer {layer_nb}: {span_dict}")

                experiment.log_metrics(log_dict, step=step)


def log_gradients(accelerator, model, experiment, step, epoch, num_training_steps):
    step = step + epoch * num_training_steps

    if experiment != None:  
        if accelerator != None:                
            if accelerator.is_main_process:
                total_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        experiment.log_metric(f"grad_norm/{name}", param_norm, step=step)
                
                total_norm = total_norm ** 0.5
                experiment.log_metric("total_grad_norm", total_norm, step=step)



def main(arg_dict):
    config_dict = arg_dict["config"]
    settings = arg_dict["settings"]
    
    enable_accelerate = settings["accelerate"]
    logging = settings["logging"]

    # Main arguments
    dataset_path = settings["dataset"]
    model_path = settings["model"]
    model_save_path = settings["save_path"]
    tokenizer_path = settings["tokenizer"]

    task = settings["task"]

    accelerator = Accelerator(log_with="comet_ml", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device 
    #device="cpu"

    task_num_labels = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "qnli": 2,
        "qqp": 2,
        "rte": 2,   
        "sst2": 2,
        "wnli": 2
    }
    
    tokenizer = get_tokenizer(tokenizer_path)
        
    print("Initializing training. ")
    config = CustomRobertaConfig.from_dict(config_dict)
    config.vocab_size = tokenizer.vocab_size
    config.num_labels = task_num_labels[task]

    model = RobertaForSequenceClassification(config=config).from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
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
                {'params': other_params, 'lr': settings["lr"]},  # Default learning rate for most parameters
                {'params': alpha_params, 'lr': settings["alpha_lr"]}          # Higher learning rate for alpha
            ], betas=(0.9, 0.98), eps=1e-6)
    elif config.attn_mechanism == "adaptive":
        if "adaptive_lr" in settings.keys():
            adaptive_params = []
            for i in range(len(model.roberta.encoder.layer)):
                adaptive_params.append(model.roberta.encoder.layer[i].attention.self.alpha)

            other_params = []
            for name, param in model.named_parameters():
                if not name.endswith('attention.self.adaptive_mask._mask.current_val'):
                    other_params.append(param)
            optim = AdamW([
                    {'params': other_params, 'lr': settings["lr"]},  # Default learning rate for most parameters
                    {'params': adaptive_params, 'lr': settings["adaptive_lr"]}          # Higher learning rate for alpha
                ], betas=(0.9, 0.98), eps=1e-6)
        else:
            optim = AdagradWithGradClip(params = model.parameters(), grad_clip=0.03, lr=0.07)
    else:
        optim = AdamW(model.parameters(), lr=settings["lr"]) # typical range is 1e-6 to 1e-4
    
    train_dataloader = get_dataloader(settings["batch_size"], dataset_path)

    # Number of training epochs and warmup steps
    epochs = settings["epochs"]
    num_training_steps = epochs * len(train_dataloader)

    num_warmup_steps = int(0.06 * num_training_steps) if "warmup_steps" not in settings.keys() else settings["warmup_steps"]
    optim_strat = "linear" if "optim_strat" not in settings.keys() else settings["optim_strat"]

    # Initialize the scheduler
    scheduler = get_scheduler(
        optim_strat, 
        optimizer=optim, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    # Accelerator function
    model, optim, dataloader, scheduler = accelerator.prepare(
        model, optim, train_dataloader, scheduler
    )
    #dataloader = train_dataloader

    print("Beginning training process. ")
    # WandB stuff
    
    if logging:
        experiment = None
        if enable_accelerate: 
            accelerator.init_trackers(project_name="benchmarkIR", config = config.to_dict())
            if accelerator.is_main_process:
                experiment  = comet_ml.Experiment(project_name="benchmarkIR", auto_metric_step_rate=100)
                experiment.set_name(f"{settings['exp_name']}_{task}")
        else:
            experiment = comet_ml.Experiment(project_name="benchmarkIR", auto_metric_step_rate=100)
            experiment.set_name(f"{settings['exp_name']}_{task}")



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

            
            if (i%100==0) & logging:
                log_metrics(accelerator, model, experiment, i, epoch, len(loop), config, loss)
                #log_gradients(accelerator, model, experiment, i, epoch, len(loop))
            
        #unwrapped_model = accelerator.unwrap_model(model)
        #unwrapped_model.save_pretrained(
        #    model_save_path, # used to be model_path
        #    is_main_process=accelerator.is_main_process,
        #    save_function=accelerator.save,
        #)
    
        if logging:
            log_metrics(accelerator, model, experiment, i, epoch, len(loop), config, loss, print_metrics=True)


    print("Training done. Saving model. ")
    accelerator.end_training()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        model_save_path, # used to be model_path
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

if __name__ == "__main__":
    args = parse_arguments()
    
    if len(args.config_dict)>0:
        original_arg_dict = json.loads(args.config_dict)
    else:   
        config_dict = os.path.join("src/lm/configs", args.config+"_finetune.json")
        with open(config_dict) as fp: 
            original_arg_dict = json.load(fp)
        
    for key in original_arg_dict["settings"]:
        if type(original_arg_dict["settings"][key]) == str:
            original_arg_dict["settings"][key] = original_arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)

    if original_arg_dict["settings"]["task"]=="glue":
        #tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]
        tasks = ["cola", "mrpc", "rte", "sst2", "wnli", "qnli"]

        for task in tasks:
            print(f"============ Processing {task} ============")

            # Adjusting the config for each task
            arg_dict = copy.deepcopy(original_arg_dict)

            try:    
                model = RobertaForSequenceClassification.from_pretrained(arg_dict["settings"]["model"])
            except:
                arg_dict["settings"]["model"] = os.path.join(arg_dict["settings"]["model"], "roberta_"+task)

            if not os.path.exists(arg_dict["settings"]["save_path"]):
                os.mkdir(arg_dict["settings"]["save_path"])

            arg_dict["settings"]["task"] = task
            
            arg_dict["settings"]["save_path"] = os.path.join(arg_dict["settings"]["save_path"], "roberta_"+task)  # Changed 'save_path' from 'model'
            
            arg_dict["settings"]["dataset"] = os.path.join(arg_dict["settings"]["dataset"], os.path.join(task, task+"_train.pt"))

            main(arg_dict)
        
    else:
        main(original_arg_dict)