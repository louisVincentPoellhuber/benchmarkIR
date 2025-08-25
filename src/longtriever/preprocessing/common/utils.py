import os
import logging
import argparse
import json
import csv
import requests
import zipfile
import io
import fitz 
import re
import wget
import gzip
import sqlite3
import subprocess
import shutil
import time

from transformers import AutoTokenizer
from accelerate import Accelerator
import torch

import pandas as pd

import ir_datasets
from tqdm import tqdm

from typing import List
from abc import ABC, abstractmethod

import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

DATASETS_PATH = STORAGE_DIR+"/datasets"

MAIN_PROCESS = Accelerator().is_main_process

USER_AGENT = "Benchmark_for_Long-Text_Retrieval_Scraper/0.0 (louis.vincent.poellhuber@umontreal.ca)"

logging.basicConfig( 
    encoding="utf-8", 
    filename="preprocessing.log", 
    filemode="a", 
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
    )

def log_message(message, level=logging.INFO, print_message = False):
    if MAIN_PROCESS:
        logging.log(msg=message, level=level)
        if print_message:
            print(message)

def parse_arguments():
    argparser = argparse.ArgumentParser("Download dataset and preprocess it.")
    argparser.add_argument('--datapath', default=STORAGE_DIR+"/datasets/vault") 
    argparser.add_argument('--overwrite', default=False) 

    args = argparser.parse_args()

    return args

def load_jsonl(filepath):
    corpus = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())  # Parse each line as a JSON object
            corpus[doc["_id"]] = doc  # Use the document ID as the key
    return corpus

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

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

class DatasetProcessor():
    def __init__(self, datapath, dataset_name, corpus_name, overwrite=False):
        self.name = dataset_name
        self.corpus_name = corpus_name

        dataset_dir, corpus_dir, corpus_download_dir, qrel_dir = self._make_folders(datapath, dataset_name, corpus_name)
        self.dataset_dir = dataset_dir
        self.corpus_dir = corpus_dir
        self.corpus_download_dir = corpus_download_dir
        self.qrel_dir = qrel_dir

        self.overwrite = overwrite

    
    def _make_folders(self, datapath, dataset_name, corpus_name):
        vault_dir = datapath
        os.makedirs(vault_dir, exist_ok=True)

        general_corpus_dir = os.path.join(vault_dir, "corpus")
        os.makedirs(general_corpus_dir, exist_ok=True)

        corpus_dir = os.path.join(general_corpus_dir, corpus_name)
        os.makedirs(corpus_dir, exist_ok=True)

        corpus_download_dir = os.path.join(corpus_dir, "downloads")
        os.makedirs(corpus_download_dir, exist_ok=True)

        dataset_dir = os.path.join(vault_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        qrel_dir = os.path.join(dataset_dir, "qrels")
        os.makedirs(qrel_dir, exist_ok=True)

        return dataset_dir, corpus_dir, corpus_download_dir, qrel_dir

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def process_corpus(self):
        pass

    @abstractmethod
    def process_queries(self):
        pass

    @abstractmethod
    def process_qrels(self):
        pass
