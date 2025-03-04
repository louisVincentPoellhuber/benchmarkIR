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
print(f"|  Is cuda available: {torch.cuda.is_available()}")
print(f"|  Device count: {torch.cuda.device_count()}")
print(f"|  Device name 0: {torch.cuda.get_device_name(0)}")
print(f"|  CUDA version: {torch.version.cuda}")
print(f"|  cuDNN version: {torch.backends.cudnn.version()}")
print("|  Default dtype:", torch.get_default_dtype())
print("|  Supported dtypes:", torch.cuda.get_device_capability())
print("|  AMP enabled:", torch.cuda.is_bf16_supported())  # Checks if BF16 is available
print("|  Default AMP dtype:", torch.get_autocast_gpu_dtype()) 

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

def main(arg_dict, accelerator):
    config_dict = arg_dict["config"]
    settings = arg_dict["settings"]

    tokenizer_path = settings["tokenizer"]
    task = settings["task"]
    enable_logging = settings["logging"]
    logging_steps = settings["logging_steps"]

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
        
    log_message("Initializing model.", logging.WARNING)
    config = CustomRobertaConfig.from_dict(config_dict)
    config.vocab_size = tokenizer.vocab_size
    config.num_labels = task_num_labels[task]

    # TODO: In modeling_utils.py, delete unused parameters
    # seed = random.randint(0, 1000000)
    # print(f"Seed: {seed}")
    # train_args = TrainingArguments(
    #     output_dir=settings["save_path"], 
    #     do_train=settings["train"],  
    #     do_eval=(settings["evaluate"]) | (settings["validate"]), # If eval or validation is True, we'll evaluate
    #     eval_strategy=settings["eval_strategy"], 
    #     per_device_train_batch_size=settings["batch_size"],
    #     learning_rate=settings["lr"], 
    #     weight_decay=settings["weight_decay"], 
    #     adam_beta1=settings["adam_beta1"], 
    #     adam_beta2=settings["adam_beta2"], 
    #     adam_epsilon=settings["adam_epsilon"],
    #     num_train_epochs=settings["epochs"], 
    #     lr_scheduler_type=settings["lr_scheduler_type"],
    #     warmup_ratio=settings["warmup_ratio"],
    #     warmup_steps=settings["warmup_steps"],
    #     save_strategy=settings["save_strategy"],
    #     save_total_limit=settings["save_total_limit"],
    #     optim=settings["optim"],
    #     auto_find_batch_size=settings["auto_find_batch_size"], # Could be interesting to switch to true
    #     resume_from_checkpoint=settings["resume_from_checkpoint"],
    #     run_name=f"{settings['exp_name']}_{task}", 
    #     report_to="comet_ml",
    #     logging_steps=settings["logging_steps"]#, 
    #     # seed=seed
    #     # max_grad_norm=100#, The grad norm is set so high to avoid gradient clipping. However, disabling it with 0 or None prevents comet from logging the Grad norm, hence the 100. 
    #     # ddp_backend="nccl",
    #     # ddp_find_unused_parameters=True
    # )

    if settings["train"]:
        
        experiment = None
        if enable_logging:
            accelerator.init_trackers(project_name="new-attention", config = config.to_dict())
            if accelerator.is_main_process:
                experiment  = comet_ml.Experiment(project_name="new-attention", auto_metric_step_rate=logging_steps)
                experiment.set_name(f"{settings['exp_name']}_{task}")

        # Load model
        model = RobertaForSequenceClassification(config=config).from_pretrained(settings["model"], config=config, ignore_mismatched_sizes=True)
        model.to(accelerator.device)
        model.train()

        train_dataset_path = os.path.join(settings["dataset_dir"], os.path.join(task, f"{task}_train.pt"))
        train_dataloader = get_dataloader(settings["batch_size"], train_dataset_path)

        validation_dataloader = None
        if settings["validate"]:
            validation_dataset_path = os.path.join(settings["dataset_dir"], os.path.join(task, f"{task}_validation.pt"))
            validation_dataloader = get_dataloader(settings["batch_size"], validation_dataset_path)
            
        # TODO : Add customizable settings
        # Can be used instead of the default parameters
        optim = AdamW(
            model.parameters(), 
            lr=settings["lr"]#, 
            # betas=(settings["adam_beta1"], settings["adam_beta2"]), 
            # eps = settings["adam_epsilon"], 
            # weight_decay=settings["weight_decay"]
            ) 

        # TODO: Add customizable settings
        epochs = settings["epochs"]
        num_training_steps = epochs * len(train_dataloader)
        num_warmup_steps = int(0.06 * num_training_steps)

        # Initialize the scheduler
        scheduler = get_scheduler(
            settings["lr_scheduler_type"], 
            optimizer=optim, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )

        
        model, optim, dataloader, scheduler = accelerator.prepare(
            model, optim, train_dataloader, scheduler
        )

    
        log_message("Beginning training process.", logging.WARNING)
        
        step = 0
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True)
            for i, batch in enumerate(loop):
                input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
                mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
                labels = batch["labels"]#.to(device)
                print(f"\n{input_ids.device}\n")
                step += 1 # Useful for logging

                # Initialize gradients
                optim.zero_grad()

                # Pass inputs through model
                outputs = model(input_ids, attention_mask = mask, labels = labels)

                # Adjust parameters according to the loss's gradient
                loss = outputs.loss
                #loss.backward() # again, replaced by the accelerator version
                accelerator.backward(loss)

                # Steps
                optim.step()
                scheduler.step()

                # Console info
                loop.set_description(f'Epoch: {epoch}')
                loop.set_postfix(loss = loss.item())

                # Logging info
                if (step % logging_steps==0) & enable_logging & accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    log_metrics(unwrapped_model, scheduler, optim, experiment, loss, step)

            # Saving the model
            if settings["save_strategy"] == "epoch":
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    settings["save_path"], 
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
        

        # Old debug stuff
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

        log_message("Training done. Saving model.", logging.WARNING)

        # Saving model
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            settings["save_path"], 
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

        # Ending modules
        accelerator.free_memory()
        del model, optim, dataloader 
        torch.cuda.empty_cache()   
        comet_ml.end()


    # TODO: Finish evaluate
    if settings["evaluate"]:
        # Load the model from where it was saved
        model = RobertaForSequenceClassification(config=config).from_pretrained(settings["save_path"], config=config, ignore_mismatched_sizes=True)
        model.eval()

        # Evaluation dataset
        eval_dataset_path = os.path.join(settings["dataset_dir"], os.path.join(task, f"{task}_test.pt"))
        eval_dataloader = get_dataloader(settings["batch_size"], eval_dataset_path, train=False)

        # Task metrics
        task_metric = evaluate.load("glue", task, trust_remote_code=True)
        accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
        metrics = {}

        log_message("Beginning evaluation process.", logging.WARNING)
        
        step = 0
        loop = tqdm(eval_dataloader, leave=True)
        for i, batch in enumerate(loop): 
            input_ids = batch["input_ids"]#.to(device) # already taken care of by Accelerator
            mask = batch["attention_mask"]#.to(device) # REMOVE COMMENTS IF U REMOVE ACCELERATOR
            labels = batch["labels"]#.to(device) 
            step +=1
            
            outputs = model(input_ids, attention_mask = mask, labels = labels)
            predictions = torch.argmax(outputs.logits, axis=1)

            task_performance = task_metric.compute(predictions=predictions, references=labels)
            accuracy_performance = accuracy_metric.compute(predictions=predictions, references=labels)

            total_performance = task_performance | accuracy_performance

            if not metrics:
                metrics = {key:[total_performance[key]] for key in total_performance.keys()}
            else:
                for key in metrics.keys():
                    metrics[key].append(total_performance[key])



        log_message("Evaluating done. Computing metrics.", logging.WARNING)

        compute_metrics(metrics, arg_dict)

        log_message("Metrics saved. Have a nice day :)\n\n\n", logging.WARNING)
    



if __name__ == "__main__":
    args = parse_arguments()
    
    if len(args.config_dict)>0:
        original_arg_dict = json.loads(args.config_dict)
    else:   
        config_dict = os.path.join("src/new-attention/configs", args.config)
        with open(config_dict) as fp: 
            original_arg_dict = json.load(fp)
        
    original_arg_dict = default_args(original_arg_dict)

    log_message(f"============ Running experiment: {original_arg_dict['settings']['exp_name']}. ============", logging.WARNING)
    log_message(f"Model Configuration: {original_arg_dict}", logging.INFO)

    accelerator = Accelerator(log_with="comet_ml", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    if not os.path.exists(original_arg_dict["settings"]["save_path"]): 
        os.mkdir(original_arg_dict["settings"]["save_path"])

    if original_arg_dict["settings"]["task"]=="glue":
        #tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]
        tasks = ["cola", "mrpc", "rte", "sst2", "wnli", "qnli"]

        #  We need the For for each task
        for task in tasks:
            log_message(f"Processing {task}.", logging.WARNING)

            # Adjusting the config for each task
            arg_dict = copy.deepcopy(original_arg_dict)

            arg_dict["settings"]["task"] = task
            arg_dict["settings"]["save_path"] = os.path.join(arg_dict["settings"]["save_path"], task)  # I create subfolder for each task
           
            if not os.path.exists(arg_dict["settings"]["save_path"]): 
                os.mkdir(arg_dict["settings"]["save_path"])
            
            main(arg_dict, accelerator)
        
    else:
        original_arg_dict["settings"]["save_path"] = os.path.join(original_arg_dict["settings"]["save_path"], original_arg_dict["settings"]["task"])  # I create subfolder for each task

        if not os.path.exists(original_arg_dict["settings"]["save_path"]): 
            os.mkdir(original_arg_dict["settings"]["save_path"])
            
        main(original_arg_dict, accelerator)