from pathlib import Path
from tqdm import tqdm
import os
import argparse

from transformers import RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import torch


def parse_arguments():
    argparser = argparse.ArgumentParser("masked language modeling")
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--input_dir', default="/part/01/Tmp/lvpoellhuber/datasets")
    argparser.add_argument('--model_dir', default="/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm")
    argparser.add_argument('--tokenizer_dir', default="/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm")
    argparser.add_argument('--batch_size', default=16) # TODO: same
    argparser.add_argument('--checkpoint', default=None)
    argparser.add_argument('--attention', default="default") # default, adaptive, bpt, dynamic
    argparser.add_argument('--logging', default=False) # default, adaptive, dtp

    args = argparser.parse_args()

    return args

def train_BPETokenizer(paths, save_dir):
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=paths, vocab_size=30522, min_frequency=2, 
                    #special_tokens=[
                    #    '<s>', '<pad>', '</s>', '<unk>', '<mask>'
                    #])
    )
    
    print("Saving tokenizer...")
    tokenizer.save_model(save_dir)
    
    return tokenizer


def create_sample_files(dataset, out_path):
    text_data = []
    file_count = 0

    print("Creating sample files.")
    # Create sample files for all the text data
    for sample in tqdm(dataset["train"]):
        sample = sample["text"].replace('\n', ' ')
        text_data.append(sample)
        if len(text_data) == 100000:
            file_path = os.path.join(out_path, f"wiki_en_{file_count}.txt")
            with open(file_path, 'w') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
    # Last sample
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(text_data))

def mlm(tensor):
    rand = torch.rand(tensor.shape)
    mask_arr = (rand < 0.15) * (tensor > 2)

    for i in range(tensor.shape[0]):
         selection = torch.flatten(mask_arr[i].nonzero()).tolist()
         tensor[i, selection] = 4
        
    return tensor

def generate_dataset(tokenizer, paths, save_path, overwrite=False):
    if os.path.exists(save_path) & ~overwrite:
        print("Loading dataset. ")
        dataset = Dataset(torch.load(save_path))
    else:
        print("Generating dataset object. ")

        input_ids = []
        mask = []
        labels = []

        for path in tqdm(paths[:1]):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
            
            sample = tokenizer(lines, max_length = 512, padding = "max_length", truncation=True, return_tensors = "pt")
            labels.append(sample.input_ids)
            mask.append(sample.attention_mask)
            input_ids.append(mlm(sample.input_ids.detach().clone()))
        
        input_ids = torch.cat(input_ids) # concatenate all the tensors
        mask = torch.cat(mask) 
        labels = torch.cat(labels) 

        encodings = {
            "input_ids": input_ids, # tokens with mask 
            "attention_mask": mask,
            "labels": labels # tokens without mask
        }

        dataset = Dataset(encodings)
        dataset.save(save_path)

    return dataset

def generate_eval_dataset(data, tokenizer, save_path, split, overwrite=False):
    if os.path.exists(save_path) & ~overwrite:
        print("Loading dataset. ")
        dataset = Dataset(torch.load(save_path))
    else:
        print("Generating dataset object. ")
        data_split = data[split]

        input_ids = []
        mask = []
        labels = []

        for i in tqdm(range(len(data_split))):
            sequence = data_split[i]["premise"] + ". " + data_split[i]["hypothesis"]
            tokenized_seq = tokenizer(sequence, max_length=512, padding="max_length", truncation=True, return_tensors = "pt")
            
            input_ids.append(tokenized_seq.input_ids)
            mask.append(tokenized_seq.attention_mask)
            labels.append(data_split[i]["label"])
        
        input_ids = torch.cat(input_ids) # concatenate all the tensors
        mask = torch.cat(mask) 
        labels = torch.Tensor(labels).long()

        encodings = {
            "input_ids": input_ids, # tokens with mask 
            "attention_mask": mask,
            "labels": labels # tokens without mask
        }

        dataset = Dataset(encodings)
        dataset.save(save_path)

    return dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    
    def __getitem__(self, i):
        return  {key: tensor[i] for key, tensor in self.encodings.items()}
    
    def save(self, save_path):
        torch.save(self.encodings, save_path)


if __name__ == "__main__":
    
    args = parse_arguments()
