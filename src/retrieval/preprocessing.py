from pathlib import Path
from tqdm import tqdm
import os
import argparse

import torch
import subprocess

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer
import json
from pyserini.search.lucene import LuceneSearcher

from modeling_utils import *

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

DATASETS_PATH = STORAGE_DIR+"/datasets"

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

def parse_arguments():
    argparser = argparse.ArgumentParser("masked language modeling")
    argparser.add_argument('--task', default="nq_bm25") # wikipedia
    argparser.add_argument('--datapath', default=STORAGE_DIR+"/datasets") 
    argparser.add_argument('--overwrite', default=False) # wikipedia

    args = argparser.parse_args()

    return args


def get_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        raise ValueError("Tokenizer not found. Please train a new tokenizer using src/preprocessing.py or provide a correct HF tokenizer.")
    
    return tokenizer


def get_pairs_dataloader(batch_size, dataset_path):
    dataset = PairsDataset(torch.load(dataset_path, weights_only=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn = custom_pairs_collate_fn)

    return dataloader

def custom_pairs_collate_fn(batch):
    queries = [item['query'] for item in batch]
    docs = [item['documents'] for item in batch]
    return {'queries': queries, 'documents': docs}

def get_triplets_dataloader(batch_size, dataset_path):
    dataset = TripletDataset(torch.load(dataset_path, weights_only=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn = custom_triplets_collate_fn)

    return dataloader

def custom_triplets_collate_fn(batch):
    queries = [item['query'] for item in batch]
    pos = [item['pos'] for item in batch]
    neg = [item['neg'][0] for item in batch] # Take only the top result after all
    docs = [[item["pos"], item["neg"][0]] for item in batch]
    return {'queries': queries, 'positive': pos, "negatives": neg, "documents":docs}

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

def preprocess_nq_pairs(out_dir):
    out_dir = os.path.join(out_dir, "nq")
    log_message(f"Pre-Processing Natural Questions.")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq-train.zip"
    data_path = util.download_and_unzip(url, out_dir)

    log_message("Loading data.")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")

    log_message("Creating train_pairs.pt file.")
    pairs_queries = []
    pairs_docs = []
    for qid in tqdm(qrels.keys()):
        query = queries[qid]
        docid = list(qrels[qid].keys())[0]
        doc = corpus[docid]

        pairs_queries.append(query)
        pairs_docs.append(doc)

    pairs = {
        "queries":pairs_queries,
        "documents":pairs_docs
    }

    dataset = PairsDataset(pairs)    
    save_path = os.path.join(out_dir,"train_pairs.pt")
    dataset.save(save_path)
    log_message("File saved.")

def preprocess_nq_bm25(out_dir, k=6):
    out_dir = os.path.join(out_dir, "nq")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq-train.zip"
    data_path = util.download_and_unzip(url, out_dir)

    log_message("Loading data.")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    
    index_dir = os.path.join(out_dir, "bm25index")
    if not os.path.exists(index_dir):
        log_message("Corpus not yet indexed: indexing.")

        log_message("Writing corpus to JSONL.")
        output_file = os.path.join(out_dir, "corpus.jsonl")
        with open(output_file, "w") as f:
            for doc_id, doc in corpus.items():
                json.dump({"id": doc_id, "title": doc["title"], "contents": doc["text"]}, f)
                f.write("\n")

        log_message("Executing Pyserini indexing command. Note: Errors might appear, but indexing should run properly regardless.")
        cmd = [
            "python", "-m", "pyserini.index",
            "-collection", "JsonCollection",
            "-input", output_file,
            "-index", index_dir,
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", "4",
            "-storePositions", "-storeDocvectors", "-storeRaw"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        log_message("Corpus already indexed.")
    
    log_message(f"Creating train_triplets.pt file. Retrieving top-{k} documents for each query.")
    searcher = LuceneSearcher(index_dir)

    triplet_queries = []
    triple_pos_docs = []
    triple_neg_docs = []
    for qid in tqdm(qrels.keys()):
        # Add query
        query = queries[qid]
        triplet_queries.append(query)
        
        # Add positive document
        pos_docid = list(qrels[qid].keys())[0]
        doc = corpus[pos_docid]
        triple_pos_docs.append(doc)

        # Get the negative documents using BM25
        # NOTE: I might wanna have the document IDs somewhere. Not sure. 
        negatives = searcher.search(query, k=k)
        neg_docs = []
        for i in range(len(negatives)):
            if (negatives[i].docid != pos_docid) & (len(neg_docs)<5):
                neg_docs.append(corpus[negatives[i].docid])
        triple_neg_docs.append(neg_docs)

    triplets = {
        "queries":triplet_queries,
        "positives":triple_pos_docs,
        "negatives":triple_neg_docs
    }

    dataset = TripletDataset(triplets)    
    save_path = os.path.join(out_dir,"train_triplets.pt")
    dataset.save(save_path)
    log_message("File saved.")

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
    if task=="nq":
        preprocess_nq_pairs(datapath)
    if task=="nq_bm25":
        preprocess_nq_bm25(datapath)
    elif task=="other_task":
        #raise ValueError("Invalid dataset. Please choose one of the following: ['wikipedia'].")
        pass

if __name__ == "__main__":
    
    args = parse_arguments()
    
    preprocess_main(args.task, args.datapath, args.overwrite)
