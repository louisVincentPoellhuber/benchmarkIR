from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from accelerate import Accelerator
from model_biencoder import BiEncoder
from custom_search import CustomFaissSearch

import os
import random
import argparse
import json

from modeling_utils import *

import torch

import dotenv
dotenv.load_dotenv()


MAIN_PROCESS = Accelerator().is_main_process
STORAGE_DIR = os.getenv("STORAGE_DIR")


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="eval_ms_biencoder")
    argparser.add_argument('--config_dict', default={}) 
    
    args = argparser.parse_args()

    return args

def evaluate_dpr(arg_dict):
    settings = arg_dict["settings"]
    config_dict = arg_dict["config"]

    # Main arguments
    dpr_path = settings["save_path"]
    batch_size = settings["batch_size"]
    task = settings["task"]
    exp_name = settings["exp_name"]

    log_note_path = os.path.join(dpr_path, "slurm_ids.txt")
    with open(log_note_path, "a") as log_file:
        slurm_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else "local"
        log_file.write(f"Evaluating Job Slurm ID: {slurm_id}; Computer: {os.uname()[1]}\n")

    log_message(f"========================= Evaluating run {exp_name}.=========================")
    
    # Are we testing a model directly from HuggingFace?
    if settings["eval_hf_model"]: 
        q_model_path = config_dict["q_model"]
        doc_model_path = config_dict["doc_model"]

        if not os.path.exists(dpr_path):
            os.mkdir(dpr_path)

        dpr_model = BiEncoder(
            model_path=(q_model_path, doc_model_path),
            normalize=config_dict["normalize"],
            prompts={"query": config_dict["query_prompt"], "passage": config_dict["passage_prompt"]},
            attn_implementation=config_dict["attn_implementation"], 
            sep = config_dict["sep"]
        )
    else: # Otherwise we assume we load it from the save path. 
        dpr_model = BiEncoder.from_pretrained(settings["save_path"])
        
    dpr_model.eval()

    # faiss_search = FlatIPFaissSearch(dpr_model, batch_size=batch_size)
    faiss_search = CustomFaissSearch(dpr_model, batch_size=batch_size, index_path=dpr_path)

    data_path = os.path.join(STORAGE_DIR, "datasets", task)
    if task=="nq": 
        data_path = os.path.join(data_path, "nq")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    if os.path.exists(os.path.join(dpr_path, "default.flat.tsv")):
        faiss_search.load(dpr_path, prefix="default")
        log_message("Already indexed, loading.")
    else:
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
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+".json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)

    for key in arg_dict["config"]:
        if type(arg_dict["config"][key]) == str:
            arg_dict["config"][key] = arg_dict["config"][key].replace("STORAGE_DIR", STORAGE_DIR)


    evaluate_dpr(arg_dict)