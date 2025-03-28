from pathlib import Path
from tqdm import tqdm
import os
import argparse
from accelerate import Accelerator
import nltk
from torch.utils.data import DataLoader
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer
import json
import logging
import dotenv
dotenv.load_dotenv()
from typing import List

STORAGE_DIR = os.getenv("STORAGE_DIR")

DATASETS_PATH = STORAGE_DIR+"/datasets"

MAIN_PROCESS = Accelerator().is_main_process

def log_message(message, level=logging.WARNING):
    if MAIN_PROCESS:
        print(message)
        logging.log(msg=message, level=level)

class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs["queries"])
    
    def __getitem__(self, i):
        query = self.pairs["queries"][i]
        doc = self.pairs["documents"][i]

        return {"query":query, "doc":doc}

    def save(self, save_path):
        torch.save(self.pairs, save_path)

class DynamicPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer_path = "google-bert/bert-base-uncased", sep = " [SEP] "):
        self.pairs = pairs
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.sep = sep
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, i):
        item = self.pairs[i]
        query = item["query"]
        document = item["document"]["title"] + self.sep + item["document"]["text"]

        return {"query":query, "document":document}
    
    def save(self, save_path):
        torch.save(self.pairs, save_path)

        
    def collate_fn(self, batch):
        queries = [item['query'] for item in batch]
        docs = [item['document'] for item in batch]

        query_inputs = self.tokenizer(queries, truncation=True, padding=True, return_tensors="pt")
        doc_inputs = self.tokenizer(docs, truncation=True, padding=True, return_tensors="pt")

        return {
            "query_inputs": query_inputs,
            "doc_inputs": doc_inputs,
        }

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
        
    def __len__(self):
        return len(self.triplets["queries"])
    
    def __getitem__(self, i):
        query = self.triplets["queries"][i]
        positives = self.triplets["positives"][i]
        negatives = self.triplets["negatives"][i]

        return {"query":query, "pos":positives, "neg":negatives}

    
    def save(self, save_path):
        torch.save(self.triplets, save_path)

class BlockDataset(torch.utils.data.Dataset): # TODO: Add to config files
    def __init__(self, pairs, tokenizer_path = "google-bert/bert-base-uncased", sep = " [SEP] ", max_block_length = 512, max_num_blocks = 4, align_right = False):
        self.pairs = pairs
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.sep = sep
        self.max_block_length = max_block_length
        self.max_num_blocks = max_num_blocks
        self.align_right = align_right
        
    def __len__(self):
        return len(self.pairs["queries"])
    
    def __getitem__(self, i):
        query = self.pairs["queries"][i]
        doc = self.pairs["documents"][i]
        
        document = doc["title"] + self.sep + doc["text"]

        return {"query":query, "document":document}

    def save(self, save_path):
        torch.save(self.pairs, save_path)

    def block_tokenize(self, string):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)
        
        block_len = self.max_block_length - self.tokenizer.num_special_tokens_to_add(False)
        input_ids_blocks = []
        attention_mask_blocks = []
        curr_block = []

        for input_ids_sent in results['input_ids']:
            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block: # This is for overflow
                input_ids_blocks.append(
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
                attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
                curr_block = []
                if len(input_ids_blocks) >= self.max_num_blocks:
                    break
            curr_block.extend(input_ids_sent)
        if len(curr_block) > 0:
            input_ids_blocks.append(
                torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
            
            attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
        
        input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id, align_right=self.align_right)
        attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0, align_right=self.align_right)
        
        return {
            "input_ids_blocks": input_ids_blocks,
            "attention_mask_blocks": attention_mask_blocks,
        }
    
    def collate_fn(self, batch):
        queries = [item['query'] for item in batch]
        docs = [item['document'] for item in batch] # untokenized documents
        
        query_inputs = self.tokenizer(queries, truncation=True, padding=True, return_tensors="pt")
        doc_inputs = {"input_ids":[], "attention_mask":[]}
        for doc in docs:
            doc_results=self.block_tokenize(doc)
            doc_inputs["input_ids"].append(doc_results['input_ids_blocks'])
            doc_inputs["attention_mask"].append(doc_results['attention_mask_blocks'])

        doc_inputs["input_ids"]=tensorize_batch(doc_inputs["input_ids"], self.tokenizer.pad_token_id) #[B,N,L]
        doc_inputs["attention_mask"]=tensorize_batch(doc_inputs["attention_mask"], 0) #[B,N,L]

        return {
            "query_inputs": query_inputs,
            "doc_inputs": doc_inputs,
        }


def get_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        raise ValueError("Tokenizer not found. Please train a new tokenizer using src/preprocessing.py or provide a correct HF tokenizer.")
    
    return tokenizer

def get_dynamic_pairs_dataloader(batch_size, dataset_path, tokenizer_path="google-bert/bert-base-uncased", sep = " [SEP] "):
    dataset = PairsDataset(torch.load(dataset_path, weights_only=True), tokenizer_path, sep)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn = dataset.collate_fn)

    return dataloader

def get_block_dataloader(batch_size, dataset_path, tokenizer_path="google-bert/bert-base-uncased", sep = " [SEP] ", max_block_length = 512, max_num_blocks = 4, align_right = False):
    dataset = BlockDataset(torch.load(dataset_path, weights_only=True), tokenizer_path, sep, max_block_length, max_num_blocks, align_right)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn = dataset.collate_fn)

    return dataloader

def get_pairs_dataloader(batch_size, dataset_path, **kwargs):
    dataset = PairsDataset(torch.load(dataset_path, weights_only=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn = custom_pairs_collate_fn, **kwargs)

    return dataloader

def get_pairs_dataset(dataset_path):
    dataset = PairsDataset(torch.load(dataset_path, weights_only=True))
    return dataset

def custom_pairs_collate_fn(batch):
    queries = [item['query'] for item in batch]
    docs = [item['doc'] for item in batch]
    return {'queries': queries, 'documents': docs}

def get_triplets_dataloader(batch_size, dataset_path):
    effective_batch_size = int(batch_size / 2)
    dataset = TripletDataset(torch.load(dataset_path, weights_only=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = effective_batch_size, shuffle=True, collate_fn = custom_triplets_collate_fn)

    return dataloader

def custom_triplets_collate_fn(batch):
    queries = [item['query'] for item in batch]
    pos = [item['pos'] for item in batch]
    neg = [item['neg'][0] for item in batch] # Take only the top result after all

    return {'queries': queries, 'positives': pos, "negatives": neg}


def tensorize_batch(sequences: List[torch.Tensor], padding_value, align_right=False) -> torch.Tensor:
    if len(sequences[0].size()) == 1:
        max_len_1 = max([s.size(0) for s in sequences])
        out_dims = (len(sequences), max_len_1)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            updated_tensor = out_tensor[i].clone()
            if align_right:
                updated_tensor[-length_1:] = tensor
            else:
                updated_tensor[:length_1] = tensor
            out_tensor[i] = updated_tensor
        return out_tensor
    elif len(sequences[0].size()) == 2:
        max_len_1 = max([s.size(0) for s in sequences])
        max_len_2 = max([s.size(1) for s in sequences])
        out_dims = (len(sequences), max_len_1, max_len_2)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            length_2 = tensor.size(1)
            updated_tensor = out_tensor[i].clone()
            if align_right:
                updated_tensor[-length_1:, -length_2:] = tensor
            else:
                updated_tensor[:length_1, :length_2] = tensor
            out_tensor[i] = updated_tensor
        return out_tensor
    elif len(sequences[0].size()) == 3:
        max_len_1 = max([s.size(0) for s in sequences])
        max_len_2 = max([s.size(1) for s in sequences])
        max_len_3 = max([s.size(2) for s in sequences])
        out_dims = (len(sequences), max_len_1, max_len_2, max_len_3)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            length_2 = tensor.size(1)
            length_3 = tensor.size(2)
            updated_tensor = out_tensor[i].clone()
            if align_right:
                updated_tensor[-length_1:, -length_2:, -length_3:] = tensor
            else:
                updated_tensor[:length_1, :length_2, :length_3] = tensor
            out_tensor[i] = updated_tensor
        return out_tensor
    else:
        raise