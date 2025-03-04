from pathlib import Path
from tqdm import tqdm
import os
import argparse

from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling, BartTokenizer, BartModel
from tokenizers import ByteLevelBPETokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Punctuation 
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk

import logging 
from modeling_utils import log_message

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

DATASETS_PATH = STORAGE_DIR+"/datasets"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    
    def __getitem__(self, i):
        return  {key: tensor[i] for key, tensor in self.encodings.items()}
    
    def save(self, save_path):
        torch.save(self.encodings, save_path)



def parse_arguments():
    argparser = argparse.ArgumentParser("masked language modeling")
    argparser.add_argument('--task', default="scrolls") # glue, scrolls
    argparser.add_argument('--datapath', default=DATASETS_PATH) 
    argparser.add_argument('--tokenizer_path', default="FacebookAI/roberta-base")
    argparser.add_argument('--train_tokenizer', default=False) # wikipedia
    argparser.add_argument('--overwrite', default=True) # wikipedia

    args = argparser.parse_args()

    return args

def get_tokenizer(tokenizer_path):
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    except:
        raise ValueError("Tokenizer not found. Please train a new tokenizer using src/preprocessing.py or provide a correct HF tokenizer.")
    
    return tokenizer

def get_dataloader(batch_size, dataset_path, train=True):
    dataset = Dataset(torch.load(dataset_path, weights_only=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=train)
    
    return dataloader


''' This function encompasses everything needed to preprocess the MNLI dataset, from GLUE. 
Input
    data_path: Where the preprocessed data should be stored. 
    tokenizer_path: Where the tokenizer is / should be stored. 
    train_tokenizer: Whether to train the tokenizer or not. 

The function has no output, but will save the two necessary files locally: the tokenizer and the dataset. 
'''
def preprocess_glue(task, data_path, tokenizer_path, overwrite):
    
    ''' Creates a dataset object. To do so, the function iterates through the files in the eval 
    data split and tokenizes all the lines found inside. It then creates a Dataset object and saves it. 

    Input
        tokenizer: The tokenizer. 
        paths: The sample files to feed to the tokenizer. 
        save_path: Where to save the dataset object. 
    '''
    def generate_two_sentence_dataset(sentence1, sentence2, data, tokenizer, save_path, split, overwrite):
        if os.path.exists(save_path) & (not overwrite):
            print("Loading dataset. ")
            dataset = Dataset(torch.load(save_path))
        else:
            print("Generating dataset object. ")
            data_split = data[split]

            input_ids = []
            mask = []
            labels = []
            sequences = []

            for i in tqdm(range(len(data_split))):
                sequence = data_split[i][sentence1] + "</s></s>" + data_split[i][sentence2]
                tokenized_seq = tokenizer(sequence, max_length=512, padding="max_length", truncation=True, return_tensors = "pt")
                
                input_ids.append(tokenized_seq.input_ids)
                mask.append(tokenized_seq.attention_mask)
                labels.append(data_split[i]["label"])
                sequences.append(sequence)

            
            input_ids = torch.cat(input_ids) # concatenate all the tensors
            mask = torch.cat(mask) 
            labels = torch.Tensor(labels).long()

            encodings = {##
                "input_ids": input_ids, # tokens with mask 
                "attention_mask": mask,
                "labels": labels, # tokens without mask
                "sequence": sequences
            }

            dataset = Dataset(encodings)
            dataset.save(save_path)

        return dataset
    
    def generate_single_sentence_dataset(sentence, data, tokenizer, save_path, split, overwrite):
        if os.path.exists(save_path) & (not overwrite):
            print("Loading dataset. ")
            dataset = Dataset(torch.load(save_path))
        else:
            print("Generating dataset object. ")
            data_split = data[split]

            input_ids = []
            mask = []
            labels = []
            sequences = []

            for i in tqdm(range(len(data_split))):
                sequence = data_split[i][sentence]
                tokenized_seq = tokenizer(sequence, max_length=512, padding="max_length", truncation=True, return_tensors = "pt")
                
                input_ids.append(tokenized_seq.input_ids)
                mask.append(tokenized_seq.attention_mask)
                labels.append(data_split[i]["label"])
                sequences.append(sequence)

            
            input_ids = torch.cat(input_ids) # concatenate all the tensors
            mask = torch.cat(mask) 
            labels = torch.Tensor(labels).long()

            encodings = {
                "input_ids": input_ids, # tokens with mask 
                "attention_mask": mask,
                "labels": labels, # tokens without mask
                "sequence": sequences
            }

            dataset = Dataset(encodings)
            dataset.save(save_path)

        return dataset

    # Download the dataset
    data = load_dataset("glue", task, cache_dir=DATASETS_PATH)
    tokenizer = get_tokenizer(tokenizer_path)

    if task=="mnli":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("premise", "hypothesis", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("premise", "hypothesis", data, tokenizer, save_path, "validation_matched", overwrite)
    elif task=="mrpc":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "test", overwrite)
    elif (task=="stsb") | (task=="wnli"):
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "validation", overwrite)
    elif task=="rte":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "validation", overwrite)
    elif task=="qnli":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("question", "sentence", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("question", "sentence", data, tokenizer, save_path, "validation", overwrite)
    elif task=="qqp":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("question1", "question2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("question1", "question2", data, tokenizer, save_path, "validation", overwrite)
    elif (task=="sst2") | (task=="cola"):
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_single_sentence_dataset("sentence", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_single_sentence_dataset("sentence", data, tokenizer, save_path, "validation", overwrite)

def get_scrolls_dataset(task, split, model, max_source_length = 1024, max_target_length = 128, padding = "max_length"):
    split_dict = preprocess_scrolls(task, model, max_source_length, max_target_length, padding)

    return split_dict[split]

def preprocess_scrolls(task, model="facebook/bart-large", max_source_length = 1024, max_target_length = 128, padding = "max_length"):
    text_column = "input"
    summary_column = "output"
    
    # Download datasets
    dataset = load_dataset("tau/scrolls", task, cache_dir=DATASETS_PATH, trust_remote_code=True)

    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(model)

    # Preprocessing function to apply to everything
    def preprocess_function(examples, tokenizer, text_column, summary_column, padding, max_source_length, max_target_length):
        inputs = examples[text_column]
        targets = examples[summary_column]

        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        if targets[0] != None:
            labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        else:  # Because the test split doesn't have labels
            labels = {"input_ids":[[None]] * len(targets)}

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset_folder = os.path.join(DATASETS_PATH, task)

    splits_dict = {}
    for split in dataset:
        log_message(f"Doing split: {split}")

        split_dataset = dataset[split] 

        column_names = split_dataset.column_names

        filepath = os.path.join(dataset_folder, f"{split}.pt")
        split_dataset = split_dataset.map(
                        preprocess_function,
                        fn_kwargs={
                            "tokenizer": tokenizer,
                            "text_column": text_column,
                            "summary_column": summary_column,
                            "padding": padding,
                            "max_source_length": max_source_length,
                            "max_target_length": max_target_length
                        },
                        batched=True,
                        remove_columns=column_names,
                        load_from_cache_file=True,
                        cache_file_name=filepath,
                        num_proc=8,
                        desc=f"Running tokenizer on {split} dataset."
                    )
        splits_dict[split] = split_dataset
        
    return splits_dict



''' Main preprocessing function. Directs which preprocessing pipeline to use. 
Input
    dataset: Which dataset to preprocess. Choice between 'wikipedia', 'mnli'
    tokenizer_path: Where to save the tokenizer. 
    train_tokenizer: Whether to train the tokenizer.  
'''
def preprocess_main(task, datapath, tokenizer_path, train_tokenizer, overwrite=False):
    if task=="glue": # Preprocess all tasks
        tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
        #tasks = ["mnli"]
        for task in tasks:
            print(f"============ Processing {task} ============")
            task_datapath = os.path.join(datapath, task)

            if not os.path.exists(task_datapath):
                print(f"Making directory: {task_datapath}")
                os.mkdir(task_datapath)

            preprocess_glue(task, task_datapath, tokenizer_path, overwrite)
    elif task=="scrolls":
        tasks = ["gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "qasper", "quality", "contract_nli"]
        
        for task in tasks:
            dataset_folder = os.path.join(DATASETS_PATH, task)
            if not os.path.exists(dataset_folder): os.mkdir(dataset_folder)
            
            preprocess_scrolls(task)
        
if __name__ == "__main__":
    
    args = parse_arguments()
    
    preprocess_main(args.task, args.datapath, args.tokenizer_path, args.train_tokenizer, args.overwrite)
