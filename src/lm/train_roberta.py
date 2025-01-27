import comet_ml

from transformers import RobertaConfig, get_scheduler,PretrainedConfig
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs
from roberta_config import CustomRobertaConfig

import argparse
#import wandb
from tqdm import tqdm
import json
import os

from preprocessing import preprocess_main, get_dataloader, get_tokenizer, get_mlm_dataloader
from model_custom_roberta import *

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="default") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) 
    argparser.add_argument('--archive', default=False) # default, adaptive, sparse
    
    args = argparser.parse_args()

    return args

def save_model(enable_accelerate, accelerator, model, save_path):
    if enable_accelerate:
    
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        roberta_path = os.path.join(save_path, "roberta_model")
        unwrapped_model.roberta.save_pretrained(
            roberta_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
    else:
        model.save_pretrained(save_path, from_pt = True)
        roberta_path = os.path.join(save_path, "roberta_model")
        model.roberta.save_pretrained(roberta_path, from_pt=True)

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



if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        if args.archive:
            config_path = os.path.join("/home/lvpoellhuber/projects/benchmarkIR/src/lm/configs_archive/old", args.config+"_mlm.json")
        else:
            config_path = os.path.join("/home/lvpoellhuber/projects/benchmarkIR/src/lm/configs", args.config+"_mlm.json")
        
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
    model_path = settings["model"]
    model_save_path = settings["save_path"]
    tokenizer_path = settings["tokenizer"]

    if enable_accelerate:
        print("Accelerate enabled. ")
        accelerator = Accelerator(log_with="comet_ml", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        device = accelerator.device 
    else:
        print("Accelerate disabled.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        accelerator=None

    tokenizer = get_tokenizer(tokenizer_path)

    if args.archive:
        dataloader = get_dataloader(settings["batch_size"], dataset_path)
    else:
        dataloader = get_mlm_dataloader(settings["batch_size"], dataset_path, tokenizer)

    # Read the config
    print("Initializing training. ")
    config = CustomRobertaConfig.from_dict(config)
    config.vocab_size = tokenizer.vocab_size+4 if args.archive else tokenizer.vocab_size
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
                {'params': other_params, 'lr': settings["lr"]},  # Default learning rate for most parameters
                {'params': alpha_params, 'lr': 10.0}          # Higher learning rate for alpha
            ], betas=(0.9, 0.98), eps=1e-6)
    else:
        optim = AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.98), eps=1e-6) # typical range is 1e-6 to 1e-4

    # Number of training epochs and warmup steps
    epochs = settings["epochs"]
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps) if args.archive else 10000

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
    if logging:
        
        experiment = None
        if enable_accelerate: 
            accelerator.init_trackers(project_name="implement-sparse", config = config.to_dict())
            if accelerator.is_main_process:
                experiment  = comet_ml.Experiment(project_name="implement-sparse", auto_metric_step_rate=100)
        else:
            experiment = comet_ml.Experiment(project_name="implement-sparse", auto_metric_step_rate=100, )
    # You can add a config here, for the experiment
   
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

            if (i%100000==0) & (i!=0):
                save_model(enable_accelerate, accelerator, model, model_save_path)
            
            if (i%100==0) & logging:
                log_metrics(accelerator, model, experiment, i, epoch, len(loop), config, loss)

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss = loss.item())
        
        save_model(enable_accelerate, accelerator, model, model_save_path)


    print("Training done. Saving model. ")
    
    accelerator.end_training()
    save_model(enable_accelerate, accelerator, model, model_save_path)
