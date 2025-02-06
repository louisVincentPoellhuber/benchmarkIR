import comet_ml

from preprocessing import *
from model_custom_roberta import *

import argparse
import pandas as pd
import json

from transformers import RobertaConfig, get_scheduler
from roberta_config import CustomRobertaConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from datasets import load_dataset, load_metric
import copy

from itertools import compress

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)


def parse_arguments():
    argparser = argparse.ArgumentParser("Evaluating Roberta")
    argparser.add_argument('--config', default="investigate")
    argparser.add_argument('--config_dict', default={}) 
    args = argparser.parse_args()

    return args

def main(arg_dict):
    roberta_config_dict = arg_dict["roberta_config"]
    adaptive_config_dict = arg_dict["adaptive_config"]
    sparse_config_dict = arg_dict["sparse_config"]
    settings = arg_dict["settings"]
    
    enable_accelerate = settings["accelerate"]
    logging = settings["logging"]
    model_name = settings["exp_name"]

    # Main arguments
    dataset_path = settings["dataset"]
    roberta_model_path = settings["roberta_model"]
    adaptive_model_path = settings["adaptive_model"]
    sparse_model_path = settings["sparse_model"]
    model_save_path = os.path.join(settings["save_path"], model_name + "_")
    tokenizer_path = settings["tokenizer"]

    task = settings["task"]

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
    accuracy = load_metric("accuracy", trust_remote_code = True)
    tokenizer = get_tokenizer(tokenizer_path)
    test_dataloader = get_dataloader(settings["batch_size"], dataset_path)
    
    roberta_config = CustomRobertaConfig.from_dict(roberta_config_dict)
    roberta_config.vocab_size = tokenizer.vocab_size
    roberta_config.num_labels = task_num_labels[task]

    roberta_model = RobertaForSequenceClassification(config=roberta_config).from_pretrained(roberta_model_path, config=roberta_config, ignore_mismatched_sizes=True)
    roberta_model.to(device)
    
    adaptive_config = CustomRobertaConfig.from_dict(adaptive_config_dict)
    adaptive_config.vocab_size = tokenizer.vocab_size
    adaptive_config.num_labels = task_num_labels[task]

    adaptive_model = RobertaForSequenceClassification(config=adaptive_config).from_pretrained(adaptive_model_path, config=adaptive_config, ignore_mismatched_sizes=True)
    adaptive_model.to(device)
    
    sparse_config = CustomRobertaConfig.from_dict(sparse_config_dict)
    sparse_config.vocab_size = tokenizer.vocab_size
    sparse_config.num_labels = task_num_labels[task]

    sparse_model = RobertaForSequenceClassification(config=sparse_config).from_pretrained(sparse_model_path, config=sparse_config, ignore_mismatched_sizes=True)
    sparse_model.to(device)

    # Accelerator function
    roberta_model, adaptive_model, sparse_model, dataloader = accelerator.prepare(
        roberta_model, adaptive_model, sparse_model, test_dataloader
    )

    loop = tqdm(dataloader, leave=True)


    roberta_metrics = []
    adaptive_metrics = []
    sparse_metrics = []

    num_labels = task_num_labels[task]
    prediction_distribution = torch.zeros(size=(4, num_labels), device=device)
    unique_labels = torch.arange(num_labels, device=device)
    ones = torch.ones(num_labels, device=device)

    adaptive_out_path = model_save_path+"adaptive_examples.csv"
    sparse_out_path = model_save_path+"sparse_examples.csv"
    for i, batch in enumerate(loop): 
        input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
        mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
        labels =batch["labels"]#.to(device) 
        sequences = [sequence.replace(";", ":") for sequence in batch["sequence"]]

        label_counts = torch.cat((labels, unique_labels)).unique(return_counts=True)
        prediction_distribution[0] += label_counts[1] - ones
        
        roberta_outputs = roberta_model(input_ids, attention_mask = mask, labels = labels)
        roberta_predictions = torch.argmax(roberta_outputs.logits, axis=1)

        roberta_pred_counts =  torch.cat((roberta_predictions, unique_labels)).unique(return_counts=True)
        prediction_distribution[1] += roberta_pred_counts[1] - ones

        adaptive_outputs = adaptive_model(input_ids, attention_mask = mask, labels = labels)
        adaptive_predictions = torch.argmax(adaptive_outputs.logits, axis=1)

        adaptive_pred_counts = torch.cat((adaptive_predictions, unique_labels)).unique(return_counts=True)
        prediction_distribution[2] += adaptive_pred_counts[1] - ones

        sparse_outputs = sparse_model(input_ids, attention_mask = mask, labels = labels)
        sparse_predictions = torch.argmax(sparse_outputs.logits, axis=1)

        sparse_pred_counts = torch.cat((sparse_predictions, unique_labels)).unique(return_counts=True)
        prediction_distribution[3] += sparse_pred_counts[1] - ones
        
        # Compute accuracy
        roberta_test_acc = accuracy.compute(predictions=roberta_predictions, references=labels)
        roberta_metrics.append(roberta_test_acc["accuracy"])
        adaptive_test_acc = accuracy.compute(predictions=adaptive_predictions, references=labels)
        adaptive_metrics.append(adaptive_test_acc["accuracy"])
        sparse_test_acc = accuracy.compute(predictions=sparse_predictions, references=labels)
        sparse_metrics.append(sparse_test_acc["accuracy"])

        # Where do adaptive models outperform?
        inverse_roberta_acc = roberta_predictions!=labels
        adaptive_acc = adaptive_predictions==labels
        sparse_acc = sparse_predictions==labels

        adaptive_idx = inverse_roberta_acc & adaptive_acc
        adaptive_examples = list(compress(sequences, adaptive_idx))
        adaptive_labels = list(compress(labels.tolist(), adaptive_idx))
        adaptive_examples = [task+";"+adaptive_examples[i]+";"+str(adaptive_labels[i])+"\n" for i in range(len(adaptive_examples))]

        with open(adaptive_out_path, "a") as adp_file:
            adp_file.writelines(adaptive_examples)
        
        sparse_idx = inverse_roberta_acc & sparse_acc
        sparse_examples = list(compress(sequences, sparse_idx))
        sparse_labels = list(compress(labels.tolist(), sparse_idx))
        sparse_examples = [task+";"+sparse_examples[i]+";"+str(sparse_labels[i])+"\n" for i in range(len(sparse_examples))]
        
        with open(sparse_out_path, "a") as spr_file:
            spr_file.writelines(sparse_examples)

    with open(model_save_path+"accuracy.csv", "a") as acc_file:
        roberta_accuracy = sum(roberta_metrics) / len(roberta_metrics)
        print(f"roberta_accuracy: {roberta_accuracy}")
        roberta_accuracy = str(roberta_accuracy) + "," + "roberta"+ ","+ task +"\n" 
        acc_file.write(roberta_accuracy)

        adaptive_accuracy = sum(adaptive_metrics) / len(adaptive_metrics)
        print(f"adaptive_accuracy: {adaptive_accuracy}")
        adaptive_accuracy = str(adaptive_accuracy) + "," + "adaptive"+ ","+ task +"\n" 
        acc_file.write(adaptive_accuracy)

        sparse_accuracy = sum(sparse_metrics) / len(sparse_metrics)
        print(f"sparse_accuracy: {sparse_accuracy}")
        sparse_accuracy = str(sparse_accuracy) + "," + "sparse"+ ","+ task +"\n" 
        acc_file.write(sparse_accuracy)

    print(model_save_path+"prediction_dist.csv")
    with open(model_save_path+"prediction_dist.csv", "a") as pred_file:
        label_row = "labels," + str(prediction_distribution[0].tolist()).strip("[").strip("]") +f",{task}\n"
        roberta_row = "roberta," + str(prediction_distribution[1].tolist()).strip("[").strip("]") +f",{task}\n"
        adaptive_row = "adaptive," + str(prediction_distribution[2].tolist()).strip("[").strip("]") +f",{task}\n"
        sparse_row = "sparse," + str(prediction_distribution[3].tolist()).strip("[").strip("]") +f",{task}\n"

        pred_file.writelines([label_row, roberta_row, adaptive_row, sparse_row])


if __name__ == "__main__":
    args = parse_arguments()

    if len(args.config_dict)>0:
        print(args.config_dict)
        original_arg_dict = json.loads(args.config_dict)
    else:   
        config_dict = os.path.join("src/lm/configs", args.config+".json")
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

            model_name = arg_dict["settings"]["exp_name"]

            if arg_dict["settings"]["roberta_model"] != "FacebookAI/roberta-base":
                arg_dict["settings"]["roberta_model"] = os.path.join(arg_dict["settings"]["roberta_model"], "roberta_"+task)
                arg_dict["settings"]["adaptive_model"] = os.path.join(arg_dict["settings"]["adaptive_model"], "roberta_"+task)
                arg_dict["settings"]["sparse_model"] = os.path.join(arg_dict["settings"]["sparse_model"], "roberta_"+task)
            
            arg_dict["settings"]["task"] = task            
            arg_dict["settings"]["dataset"] = os.path.join(arg_dict["settings"]["dataset"], os.path.join(task, task+"_test.pt"))

            main(arg_dict)
        
    else:
        main(original_arg_dict)