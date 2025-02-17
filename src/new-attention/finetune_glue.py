import comet_ml

from modeling_utils import *

import json
import random
from transformers import get_scheduler
from roberta_config import CustomRobertaConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import Trainer, TrainingArguments
from transformers.integrations import CometCallback
from torch.optim import AdamW
import copy
import evaluate
import numpy as np
from dataclasses import replace
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


job_computer = os.getenv("SLURM_NODELIST")
if job_computer == None: job_computer = "local"
log_message(f"Computer: {job_computer}")
log_message(f"Slurm Job ID: {JOBID}")

def main(arg_dict):
    config_dict = arg_dict["config"]
    settings = arg_dict["settings"]

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

    # seed = random.randint(0, 1000000)
    # print(f"Seed: {seed}")

    train_args = TrainingArguments(
        output_dir=settings["save_path"], 
        do_train=settings["train"],  
        do_eval=(settings["evaluate"]) | (settings["validate"]), # If eval or validation is True, we'll evaluate
        eval_strategy=settings["eval_strategy"], 
        per_device_train_batch_size=settings["batch_size"],
        learning_rate=settings["lr"], 
        weight_decay=settings["weight_decay"], 
        adam_beta1=settings["adam_beta1"], 
        adam_beta2=settings["adam_beta2"], 
        adam_epsilon=settings["adam_epsilon"],
        num_train_epochs=settings["epochs"], 
        lr_scheduler_type=settings["lr_scheduler_type"],
        warmup_ratio=settings["warmup_ratio"],
        warmup_steps=settings["warmup_steps"],
        save_strategy=settings["save_strategy"],
        save_total_limit=settings["save_total_limit"],
        optim=settings["optim"],
        auto_find_batch_size=settings["auto_find_batch_size"], # Could be interesting to switch to true
        resume_from_checkpoint=settings["resume_from_checkpoint"],
        run_name=f"{settings['exp_name']}_{task}", 
        report_to="comet_ml",
        logging_steps=5, 
        # seed=seed
        # max_grad_norm=100#, The grad norm is set so high to avoid gradient clipping. However, disabling it with 0 or None prevents comet from logging the Grad norm, hence the 100. 
        # ddp_backend="nccl",
        # ddp_find_unused_parameters=True

    )

    
    def compute_glue_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        task_metric = evaluate.load("glue", task, trust_remote_code=True)
        task_performance = task_metric.compute(predictions=predictions, references=labels)

        accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
        accuracy_performance = accuracy_metric.compute(predictions=predictions, references=labels)

        total_performance = task_performance | accuracy_performance
        return total_performance


    if settings["train"]:
        model = RobertaForSequenceClassification(config=config).from_pretrained(settings["model"], config=config, ignore_mismatched_sizes=True)

        train_dataset_path = os.path.join(settings["dataset_dir"], os.path.join(task, f"{task}_train.pt"))
        train_dataset = TextClassificationDataset(torch.load(train_dataset_path, weights_only=True))
        validation_dataset = None

        if settings["validate"]:
            validation_dataset_path = os.path.join(settings["dataset_dir"], os.path.join(task, f"{task}_validation.pt"))
            validation_dataset = TextClassificationDataset(torch.load(validation_dataset_path, weights_only=True))
            
        
        # Can be used instead of the default parameters
        # optim = AdamW(
        #     model.parameters(), 
        #     lr=settings["lr"]#, 
        #     # betas=(settings["adam_beta1"], settings["adam_beta2"]), 
        #     # eps = settings["adam_epsilon"], 
        #     # weight_decay=settings["weight_decay"]
        #     ) 

        # num_training_steps = settings["epochs"] * int((train_dataset.__len__() + settings["batch_size"]) / settings["batch_size"]) # the 4 is for the number of processes
        # num_warmup_steps = int(0.06 * num_training_steps)
        # scheduler = get_scheduler(
        #     settings["lr_scheduler_type"], 
        #    #"linear",
        #     optimizer=optim, 
        #     num_training_steps=num_training_steps,
        #     num_warmup_steps=num_warmup_steps
        # )

        trainer = Trainer(
            model = model, 
            args = train_args, 
            train_dataset=train_dataset, 
            eval_dataset=validation_dataset, 
            compute_metrics=compute_glue_metrics#, 
            # optimizers=(optim, scheduler)
        )
    

        
        print("Beginning training process.")
        log_message("Beginning training process.", logging.WARNING)
        
        train_output = trainer.train()
        
        # model = trainer.model
        # optim = trainer.optimizer
        # scheduler = trainer.lr_scheduler

      
        # print("\n\n\n\n\nConfigs\n\nModel\n\n")
        # print(model)
        # print("\n\nOptimizer\n\n")
        # print(optim)
        # print("\n\nScheduler\n\n")
        # getlr = scheduler.get_lr()
        # print(f"get lr: {getlr}")
        # getlastlr = scheduler.get_last_lr()
        # print(f"get last lr: {getlastlr}")
        # lastepoch = scheduler.last_epoch
        # print(f"last epoch: {lastepoch}")
        # kw = scheduler.lr_lambdas[0].keywords
        # print(f"lr lambdas: {kw}")
        # print("\n\n\n\n\n")


        print("Training done. Saving model.")
        log_message("Training done. Saving model.", logging.WARNING)
        model = trainer.model
        model.save_pretrained(settings["save_path"])

        comet_ml.end()


    if settings["evaluate"]:
        model = RobertaForSequenceClassification(config=config).from_pretrained(settings["save_path"], config=config, ignore_mismatched_sizes=True)

        test_dataset_path = os.path.join(settings["dataset_dir"], os.path.join(task, f"{task}_test.pt"))
        eval_dataset = TextClassificationDataset(torch.load(test_dataset_path, weights_only=True))

        train_args = replace(train_args, report_to="none")

        evaluator = Trainer(
            model = model, 
            args = train_args, 
            eval_dataset=eval_dataset, 
            compute_metrics=compute_glue_metrics
        )

        print("Beginning evaluating process.")
        log_message("Beginning training process.", logging.WARNING)
        
        eval_output = evaluator.evaluate()

        print("Evaluating done. Computing metrics.")
        log_message("Training done. Saving model.", logging.WARNING)

        compute_metrics(eval_output, arg_dict)

        print("Metrics saved. Have a nice day :)")
    



if __name__ == "__main__":
    args = parse_arguments()
    
    if len(args.config_dict)>0:
        original_arg_dict = json.loads(args.config_dict)
    else:   
        config_dict = os.path.join("src/new-attention/configs", args.config)
        with open(config_dict) as fp: 
            original_arg_dict = json.load(fp)
        
    original_arg_dict = default_args(original_arg_dict)

    log_message(f"============ Finetuning {original_arg_dict['settings']['exp_name']}. ============", logging.WARNING)
    log_message(f"Model Configuration: {original_arg_dict}", logging.INFO)

    if not os.path.exists(original_arg_dict["settings"]["save_path"]): 
        os.mkdir(original_arg_dict["settings"]["save_path"])

    if original_arg_dict["settings"]["task"]=="glue":
        #tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]
        tasks = ["cola", "mrpc", "rte", "sst2", "wnli", "qnli"]

        #  We need the For for each task
        for task in tasks:
            print(f"============ Processing {task} ============")
            log_message(f"Processing {task}.", logging.WARNING)

            # Adjusting the config for each task
            arg_dict = copy.deepcopy(original_arg_dict)

            arg_dict["settings"]["task"] = task
            arg_dict["settings"]["save_path"] = os.path.join(arg_dict["settings"]["save_path"], task)  # I create subfolder for each task
           
            if not os.path.exists(arg_dict["settings"]["save_path"]): 
                os.mkdir(arg_dict["settings"]["save_path"])
            
            main(arg_dict)
        
    else:
        original_arg_dict["settings"]["save_path"] = os.path.join(original_arg_dict["settings"]["save_path"], original_arg_dict["settings"]["task"])  # I create subfolder for each task

        if not os.path.exists(original_arg_dict["settings"]["save_path"]): 
            os.mkdir(original_arg_dict["settings"]["save_path"])
            
        main(original_arg_dict)