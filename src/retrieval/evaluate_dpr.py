from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from accelerate import Accelerator
from beir.retrieval.models import SentenceBERT
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaTokenizer

from model_custom_dpr import CustomDPR, CustomRobertaModel, CustomRobertaConfig
from dpr_config import CustomDPRConfig
from model_biencoder import BiEncoder

import os
import random
import argparse
import json

from modeling_utils import *

import torch

import dotenv
dotenv.load_dotenv()

AutoConfig.register("custom-roberta", CustomRobertaConfig)
AutoModel.register(CustomRobertaConfig, CustomRobertaModel)
AutoTokenizer.register(CustomRobertaConfig, RobertaTokenizer)


MAIN_PROCESS = Accelerator().is_main_process
STORAGE_DIR = os.getenv("STORAGE_DIR")


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="default") # default, adaptive, sparse
    argparser.add_argument('--config_dict', default={}) 
    
    args = argparser.parse_args()

    return args

def evaluate_dpr(arg_dict):
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    device = "cpu"

    settings = arg_dict["settings"]

    # Main arguments
    dpr_path = settings["save_path"]
    batch_size = settings["batch_size"]
    task = settings["task"]
    
    q_model_path = settings["q_model"]
    doc_model_path = settings["doc_model"]

    if not os.path.exists(dpr_path):
        os.mkdir(dpr_path)
    
    # dpr_model = CustomDPR.from_pretrained(model_path=dpr_path, device=device)
    #dpr_model = SentenceBERT(model_path = (q_model_path, doc_model_path), sep=" [SEP] ")
    dpr_model = BiEncoder(model_path = (q_model_path, doc_model_path), sep=" [SEP] ", prompts={"query": "query: ", "passage": "passage: "})
    dpr_model.eval()
    
    faiss_search = FlatIPFaissSearch(dpr_model, batch_size=batch_size)

    log_message(f"========================= Running task {task}.=========================")
    data_path = "/Tmp/lvpoellhuber/datasets/nq/nq"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #corpus = dict(list(corpus.items())[0:100])
    if faiss_search.faiss_index == None:
        log_message("Indexing.")
        faiss_search.index(corpus=corpus)
        log_message("Saving.")
        faiss_search.save(dpr_path, prefix="default")

    retriever = EvaluateRetrieval(faiss_search, score_function="dot")

    log_message("Retrieving.")
    results = retriever.retrieve(corpus, queries)

    log_message("Evaluating.")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    metrics_path = os.path.join(dpr_path, f"{task}_metrics.txt")
    with open(metrics_path, "w") as metrics_file:
        metrics_file.write("Retriever evaluation for k in: {}".format(retriever.k_values))
        metrics_file.write(f"\nNDCG: {ndcg}\nRecall: {recall}\nPrecision: {precision}\n")

        top_k = 10

        query_id, ranking_scores = random.choice(list(results.items()))
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        metrics_file.write("Query : %s\n" % queries[query_id])

        for rank in range(top_k):
            doc_id = scores_sorted[rank][0]
            # Format: Rank x: ID [Title] Body
            metrics_file.write("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


if __name__ == "__main__":    
    
    args = parse_arguments()

    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+"_eval.json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)

    evaluate_dpr(arg_dict)