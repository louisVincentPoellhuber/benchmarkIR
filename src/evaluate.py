from preprocessing import *
from models import *

import argparse
import pandas as pd

from transformers import RobertaConfig, get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from datasets import load_dataset, load_metric

def parse_arguments():
    argparser = argparse.ArgumentParser("masked language modeling")
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--input_dir', default="/part/01/Tmp/lvpoellhuber/datasets")
    argparser.add_argument('--model_dir', default="/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_glue")
    argparser.add_argument('--batch_size', default=16) # TODO: same
    argparser.add_argument('--checkpoint', default="/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm/checkpoints")
    args = argparser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    data_path = args.input_dir
    dataset_path = os.path.join(os.path.join(data_path, "glue"), "mnli")
    model_path = args.model_dir
    chkpt_path = os.path.join(os.path.join(model_path, "roberta_glue"), "checkpoints")

    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device 
    # Get data
    data = load_dataset("glue", "mnli", cache_dir=data_path)
    metric = load_metric("glue", "mnli", trust_remote_code=True)
    
    tokenizer = RobertaTokenizerFast.from_pretrained("/part/01/Tmp/lvpoellhuber/models/custom_roberta")

    if args.do_train:   
        eval_path = os.path.join(dataset_path, "mnli_train.pt")
        dataset_train = generate_eval_dataset(data, tokenizer, eval_path, "train")
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = args.batch_size, shuffle=True)

        
        print("Initializing training. ")
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size+4, 
            max_position_embeddings=514, 
            hidden_size=768,
            num_attention_heads=12, 
            num_hidden_layers=6, # Default is 12, 6 makes training shorter
            type_vocab_size=1, 
            num_labels=4
        )

        model = RobertaForSequenceClassification(config=config).from_pretrained("/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm", config=config)
        model.to(device)

        optim = AdamW(model.parameters(), lr=1e-5) # typical range is 1e-6 to 1e-4

        # Number of training epochs and warmup steps
        epochs = 2
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

        print("Beginnig training process. ")
        # WandB stuff
        #wandb.login(key=os.getenv("WANDB_KEY"))
        #run = wandb.init(
        #    project = "benchmarkIR",
        #    config = vars(args)
        #)

        # You can add a config here, for the experiment
        accelerator.init_trackers("benchmarkIR")

        #accelerator.print(f"Resumed from checkpoint: {args.checkpoint}")
        #accelerator.load_state(args.checkpoint, strict=False)

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

    if args.do_eval:    
        eval_path = os.path.join(dataset_path, "mnli_val.pt")
        dataset_val = generate_eval_dataset(data, tokenizer, eval_path, "validation_matched")
        val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size = args.batch_size, shuffle=True)

        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size+4, 
            max_position_embeddings=514, 
            hidden_size=768,
            num_attention_heads=12, 
            num_hidden_layers=6, # Default is 12, 6 makes training shorter
            type_vocab_size=1, 
            num_labels=4
        )

        model = RobertaForSequenceClassification(config=config).from_pretrained("/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_glue", config=config)
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
        