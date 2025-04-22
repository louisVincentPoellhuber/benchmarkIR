from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from accelerate import Accelerator, DistributedDataParallelKwargs
from modeling_retriever import LongtrieverRetriever
from modeling_longtriever import Longtriever

import os
import random
import argparse
import json
import logging
import time
import dotenv
dotenv.load_dotenv()

MAIN_PROCESS = Accelerator().is_main_process
STORAGE_DIR = os.getenv("STORAGE_DIR")
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


def log_message(message, level=logging.WARNING, force_message = False):
    if force_message or MAIN_PROCESS:
        # print(message)
        logging.log(msg=message, level=level)

def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="longtriever")
    argparser.add_argument('--config_dict', default={}) 
    argparser.add_argument('--eval_batch_size', default=12) 
    
    args = argparser.parse_args()

    return args

def evaluate_dpr(arg_dict):
    settings = arg_dict["settings"]

    # Main arguments
    dpr_path = settings["save_path"]
    batch_size = settings["batch_size"]
    task = settings["task"]
    exp_name = settings["exp_name"]
    
    log_note_path = os.path.join(dpr_path, "slurm_ids.txt")
    with open(log_note_path, "a") as log_file:
        log_file.write(f"Evaluating Job ID: {JOBID}; Computer: {os.uname()[1]}\n")

    log_message(f"========================= Evaluating run {exp_name}.=========================")

    dpr_model = LongtrieverRetriever(Longtriever.from_pretrained(dpr_path))
        
    dpr_model.eval()

    faiss_search = FlatIPFaissSearch(dpr_model, batch_size=batch_size)

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
    log_message("Sleeping for 1 hours (3600s)...")
    time.sleep(3600)
    
    args = parse_arguments()

    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/longtriever/configs", args.config+".json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)

    arg_dict["settings"]["batch_size"] = args.eval_batch_size

    evaluate_dpr(arg_dict)