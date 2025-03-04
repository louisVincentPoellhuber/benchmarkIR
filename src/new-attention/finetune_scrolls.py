import comet_ml

from modeling_utils import *
from preprocessing import get_scrolls_dataset

import json
import nltk
from transformers import get_scheduler, BartTokenizer, BartForConditionalGeneration, BartConfig, DataCollatorForSeq2Seq
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
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
    data_args = arg_dict["data"]

    tokenizer_path = settings["tokenizer"]
    task = settings["task"]
    max_source_length = data_args["max_source_length"]
    max_target_length = data_args["max_target_length"]
    padding = "max_length" if data_args["pad_to_max_length"] else False

    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        
    log_message("Initializing model.", logging.WARNING)

    config = BartConfig.from_pretrained(settings["model"])
    # NOTE: Go back to .from_dict when using the config file. Otherwise I'll just use the normal BART config. 
    # config = BartConfig.from_dict(config_dict)
    # config.vocab_size = tokenizer.vocab_size

    # seed = random.randint(0, 1000000)
    # print(f"Seed: {seed}")

    train_args = Seq2SeqTrainingArguments(
        output_dir=settings["save_path"], 
        do_train=settings["train"],  
        do_eval=(settings["evaluate"]) | (settings["validate"]), # If eval or validation is True, we'll evaluate
        eval_strategy=settings["eval_strategy"], 
        per_device_train_batch_size=settings["train_batch_size"],
        per_device_eval_batch_size=settings["eval_batch_size"],
        eval_accumulation_steps=5,
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
        auto_find_batch_size=True, # Could be interesting to switch to true
        resume_from_checkpoint=settings["resume_from_checkpoint"],
        run_name=f"{settings['exp_name']}_{task}", 
        report_to="comet_ml",
        logging_steps=10, 
        generation_max_length=max_target_length,
        # seed=seed
        # max_grad_norm=100#, The grad norm is set so high to avoid gradient clipping. However, disabling it with 0 or None prevents comet from logging the Grad norm, hence the 100. 
        # ddp_backend="nccl",
        # ddp_find_unused_parameters=True
    )

    if settings["train"]:
        model = BartForConditionalGeneration(config=config).from_pretrained(settings["model"], config=config, ignore_mismatched_sizes=True)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        train_dataset = get_scrolls_dataset(task, "train", tokenizer_path, max_source_length, max_target_length, padding)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model
        )
        
        trainer = Seq2SeqTrainer(
            model = model, 
            args = train_args, 
            train_dataset=train_dataset,  
            processing_class=tokenizer,
            data_collator=data_collator
        )
        
        log_message("Beginning training process.", logging.WARNING)
        
        train_output = trainer.train()

        log_message("Training done. Saving model.", logging.WARNING)
        model = trainer.model
        model.save_pretrained(settings["save_path"])

        comet_ml.end()

    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    # NOTE: I can probably directly use this compute_metrics function. 
    def compute_scrolls_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        # NOTE: They have to decode and postprocess the data to run the metric.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # NOTE: Rouge probably needs to get raw text as input. 
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    if settings["evaluate"]:
        model = BartForConditionalGeneration(config=config).from_pretrained(settings["save_path"], config=config, ignore_mismatched_sizes=True)

        # NOTE: This is the VALIDATION dataset. I'm only using this for eyeballing, actual evaluation should be done with SCROLLS's website. 
        eval_dataset = get_scrolls_dataset(task, "validation", tokenizer_path, max_source_length, max_target_length, padding)

        train_args = replace(train_args, report_to="none")

        evaluator = Seq2SeqTrainer(
            model = model, 
            args = train_args, 
            eval_dataset=eval_dataset, 
            compute_metrics=compute_scrolls_metrics
        )

        log_message("Beginning evaluating process.", logging.WARNING)
        
        eval_output = evaluator.evaluate()

        log_message("Evaluating done. Computing metrics.", logging.WARNING)

        compute_metrics(eval_output, arg_dict)

        log_message("Metrics saved. Have a nice day :)\n", logging.WARNING)
    



if __name__ == "__main__":
    args = parse_arguments()
    
    if len(args.config_dict)>0:
        original_arg_dict = json.loads(args.config_dict)
    else:   
        config_dict = os.path.join("src/new-attention/configs", args.config)
        with open(config_dict) as fp: 
            original_arg_dict = json.load(fp)
        
    # TODO: adjust default args for GLUE vs SCROLLS. 
    original_arg_dict = default_args(original_arg_dict)

    log_message(f"============ Running experiment: {original_arg_dict['settings']['exp_name']}. ============", logging.WARNING)
    log_message(f"Model Configuration: {original_arg_dict}", logging.INFO)

    if not os.path.exists(original_arg_dict["settings"]["save_path"]): 
        os.mkdir(original_arg_dict["settings"]["save_path"])

    if original_arg_dict["settings"]["task"]=="scrolls":
        tasks = ["gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "qasper", "quality", "contract_nli"]

        #  We need the For for each task
        for task in tasks:
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