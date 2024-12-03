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

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)


def parse_arguments():
    argparser = argparse.ArgumentParser("Evaluating Roberta")
    argparser.add_argument('--config', default="roberta_glue") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) 
    args = argparser.parse_args()

    return args

def main(arg_dict):
    
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
    num_labels = task_num_labels[task]

    accelerator = Accelerator(log_with="comet_ml", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device 
    metric = load_metric("glue", task, trust_remote_code=True)
    accuracy = load_metric("accuracy", trust_remote_code = True)
    tokenizer = get_tokenizer(tokenizer_path)
    
    test_dataloader = get_dataloader(settings["batch_size"], dataset_path)
    

    config = CustomRobertaConfig.from_dict(config_dict)
    config.vocab_size = tokenizer.vocab_size
    config.num_labels = num_labels
    print(config.num_labels)

    model = RobertaForSequenceClassification(config=config).from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    model.to(device)

    # Accelerator function
    model, dataloader = accelerator.prepare(
        model, test_dataloader
    )

    loop = tqdm(dataloader, leave=True)
    metrics_df = []
    prediction_distribution = torch.zeros(size=(2, num_labels), device=device)
    unique_labels = torch.arange(num_labels, device=device)
    ones = torch.ones(num_labels, device=device)

    for i, batch in enumerate(loop): 
        input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
        mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
        labels =batch["labels"]#.to(device) 
        
        outputs = model(input_ids, attention_mask = mask, labels = labels)
        predictions = torch.argmax(outputs.logits, axis=1)
        metrics = metric.compute(predictions=predictions, references=labels)

        
        label_counts = torch.cat((labels, unique_labels)).unique(return_counts=True)
        prediction_distribution[0] += label_counts[1] - ones
        pred_counts =  torch.cat((predictions, unique_labels)).unique(return_counts=True)
        prediction_distribution[1] += pred_counts[1] - ones
        
        row = [float(outputs.loss)] + list(metrics.values())
        row_names = ["loss"] + list(metrics.keys())

        if "accuracy" not in row_names:
            train_acc = accuracy.compute(predictions=predictions, references=labels)
            row.append(train_acc["accuracy"])
            row_names.append("accuracy")
        metrics_df.append(row)
    
    print(os.path.join(model_save_path, "metrics.csv"))
    metrics_df = pd.DataFrame(metrics_df, columns = row_names)
    metrics_df.to_csv(os.path.join(model_save_path, "metrics.csv"))
    
    with open(model_save_path+"prediction_dist.csv", "a") as pred_file:
        label_row = "labels," + str(prediction_distribution[0].tolist()).strip("[").strip("]") +f",{task}\n"
        pred_row = settings["exp_name"] + "," + str(prediction_distribution[1].tolist()).strip("[").strip("]") +f",{task}\n"
        pred_file.writelines([label_row, pred_row])

if __name__ == "__main__":
    args = parse_arguments()

    if len(args.config_dict)>0:
        print(args.config_dict)
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
        tasks = ["cola", "mrpc", "qnli", "rte", "sst2", "wnli"]

        for task in tasks:
            print(f"============ Processing {task} ============")

            # Adjusting the config for each task
            arg_dict = copy.deepcopy(original_arg_dict)

            model_name = arg_dict["settings"]["exp_name"].split("_")[0]

            if arg_dict["settings"]["model"] != "FacebookAI/roberta-base":
                arg_dict["settings"]["model"] = os.path.join(arg_dict["settings"]["model"], "roberta_"+task)
            else:
                for temp_task in tasks:
                    save_path = os.path.join(arg_dict["settings"]["save_path"], "roberta_"+temp_task)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

            arg_dict["settings"]["task"] = task
            arg_dict["settings"]["save_path"] = os.path.join(arg_dict["settings"]["save_path"], "roberta_"+task)  # Changed 'save_path' from 'model'
            
            arg_dict["settings"]["dataset"] = os.path.join(arg_dict["settings"]["dataset"], os.path.join(task, task+"_test.pt"))

            main(arg_dict)
        
    else:
        main(original_arg_dict)