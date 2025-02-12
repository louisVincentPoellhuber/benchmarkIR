import comet_ml

from modeling_utils import *

import json

from transformers import get_scheduler
from roberta_config import CustomRobertaConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import Trainer, TrainingArguments
from transformers.integrations import CometCallback
from torch.optim import AdamW
import copy

import logging
JOBID = os.getenv("SLURM_JOB_ID")
if JOBID == None: JOBID = "local"
logging.basicConfig( 
    encoding="utf-8", 
    filename=f"slurm-{JOBID}.log", 
    filemode="a", 
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
    )

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)
logging.debug(f"Saving to {STORAGE_DIR}.")


def main(arg_dict):
    config_dict = arg_dict["config"]
    settings = arg_dict["settings"]
    
    enable_accelerate = settings["accelerate"]
    enable_logging = settings["logging"]

    # Main arguments
    dataset_path = settings["dataset"]
    model_path = settings["model"]
    model_save_path = settings["save_path"]
    tokenizer_path = settings["tokenizer"]

    task = settings["task"]

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
        
    print("Initializing training.")
    #log_message("Initializing training.", logging.WARNING, accelerator)
    config = CustomRobertaConfig.from_dict(config_dict)
    config.vocab_size = tokenizer.vocab_size
    config.num_labels = task_num_labels[task]

    model = RobertaForSequenceClassification(config=config).from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)

    dataset = TextClassificationDataset(torch.load(dataset_path))

    train_args = TrainingArguments(
        output_dir=model_save_path, 
        do_train=True, 
        per_device_train_batch_size=settings["batch_size"],
        learning_rate=settings["lr"], 
        weight_decay=0, 
        adam_beta1=0.9, 
        adam_beta2=0.98, 
        num_train_epochs=settings["epochs"], 
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        save_strategy="no",
        optim="adamw_torch",
        auto_find_batch_size=False, # Could be interesting to switch to true
        run_name=f"{settings['exp_name']}_{task}"
    )

    trainer = Trainer(
        model = model, 
        args = train_args, 
        train_dataset=dataset         
    )

    print("Beginning training process.")
    #log_message("Beginning training process.", logging.WARNING, accelerator)
    
    trainer.train()

    print("Training done. Saving model.")
    #log_message("Training done. Saving model.", logging.WARNING, accelerator)
    

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

    log_message(f"============ Finetuning {original_arg_dict['settings']['exp_name']}. ============", logging.WARNING, Accelerator())
    log_message(f"Model Configuration: {original_arg_dict}", logging.INFO, Accelerator())

    if original_arg_dict["settings"]["task"]=="glue":
        #tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]
        tasks = ["cola", "mrpc", "rte", "sst2", "wnli", "qnli"]

        for task in tasks:
            print(f"============ Processing {task} ============")
            log_message(f"Processing {task}.", logging.WARNING, Accelerator())

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