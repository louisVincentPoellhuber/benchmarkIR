from pathlib import Path
from tqdm import tqdm
import os
import argparse

import torch

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import RobertaTokenizerFast


import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)


DATASETS_PATH = STORAGE_DIR+"/datasets"

class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs["documents"])
    
    def __getitem__(self, i):
        query = self.pairs["queries"][i]
        doc = self.pairs["documents"][i]

        return {"query":query, "doc":doc}

    
    def save(self, save_path):
        torch.save(self.pairs, save_path)

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, queries, qrels):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, i):
        query = self.pairs["queries"][i]
        doc = self.pairs["documents"][i]

        return {"query":query, "doc":doc}
    
    def save(self, save_path):
        torch.save(self.pairs, save_path)

def parse_arguments():
    argparser = argparse.ArgumentParser("masked language modeling")
    argparser.add_argument('--task', default="dprqa") # wikipedia
    argparser.add_argument('--datapath', default=STORAGE_DIR+"/datasets") 
    argparser.add_argument('--overwrite', default=False) # wikipedia

    args = argparser.parse_args()

    return args


def get_tokenizer(tokenizer_path):
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    except:
        raise ValueError("Tokenizer not found. Please train a new tokenizer using src/preprocessing.py or provide a correct HF tokenizer.")
    
    return tokenizer


def get_dataloader(batch_size, dataset_path):
    dataset = PairsDataset(torch.load(dataset_path))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn = custom_collate_fn)

    return dataloader

def custom_collate_fn(batch):
    queries = [item['query'] for item in batch]
    docs = [item['doc'] for item in batch]
    return {'queries': queries, 'docs': docs}

def preprocess_hotpotqa(out_dir, split="train"):
    dataset_name = "hotpotqa"

    #### Download NFCorpus dataset and unzip the dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    pairs_queries = []
    pairs_docs = []
    for qid in qrels.keys():
        query = queries[qid]
        documents = qrels[qid].keys()

        for docid in documents:
            doc = corpus[docid]

            pairs_queries.append(query)
            pairs_docs.append(doc)

    pairs = {
        "queries":pairs_queries,
        "documents":pairs_docs
    }

    dataset = PairsDataset(pairs)    
    save_path = os.path.join(out_dir, os.path.join(dataset_name, split+".pt"))
    dataset.save(save_path)

def preprocess_dprqa(out_dir):
    datasets = ["msmarco", "quora", "hotpotqa", "nq-train"]

    pairs_queries = []
    pairs_docs = []

    for dataset_name in tqdm(datasets):
        #### Download NFCorpus dataset and unzip the dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, out_dir)
    
    for dataset_name in tqdm(datasets):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, out_dir)
        split = "dev" if dataset_name=="quora" else "train"
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

        for qid in qrels.keys():
            query = queries[qid]
            documents = qrels[qid].keys()

            for docid in documents:
                doc = corpus[docid]

                pairs_queries.append(query)
                pairs_docs.append(doc)
    
    
    pairs = {
        "queries":pairs_queries,
        "documents":pairs_docs
    }

    dataset = PairsDataset(pairs)    

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    save_path = os.path.join(out_dir, os.path.join("dprqa", "train.pt"))
    dataset.save(save_path)


''' Main preprocessing function. Directs which preprocessing pipeline to use. 
Input
    dataset: Which dataset to preprocess. Choice between 'wikipedia', 'mnli'
    tokenizer_path: Where to save the tokenizer. 
    train_tokenizer: Whether to train the tokenizer.  
'''
def preprocess_main(task, datapath, overwrite=False):
    if task=="hotpotqa":
        preprocess_hotpotqa(datapath)
    if task=="dprqa":
        preprocess_dprqa(datapath)
    elif task=="other_task":
        #raise ValueError("Invalid dataset. Please choose one of the following: ['wikipedia'].")
        pass

if __name__ == "__main__":
    
    args = parse_arguments()
    
    preprocess_main(args.task, args.datapath, args.overwrite)
