from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from dres import DenseRetrievalExactSearch as DRES
from accelerate import Accelerator, DistributedDataParallelKwargs

import logging
import pathlib, os
import random
import argparse
import json

import torch

#### Just some code to print debug information to stdout
logging.basicConfig(filename="src/retrieval/evaluate_dpr.log", 
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    level=logging.INFO, 
                    force=True)
#### /print debug information to stdout

from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaTokenizer

from model_custom_dpr import CustomDPR, CustomRobertaModel, CustomRobertaConfig
from dpr_config import CustomDPRConfig

#tokenizer = RobertaTokenizerFast.from_pretrained("/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm")

AutoConfig.register("custom-roberta", CustomRobertaConfig)
AutoModel.register(CustomRobertaConfig, CustomRobertaModel)
AutoTokenizer.register(CustomRobertaConfig, RobertaTokenizer)

def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="default") # default, adaptive, sparse
    
    args = argparser.parse_args()

    return args


if __name__ == "__main__":    
    args = parse_arguments()
    config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR/src/retrieval/configs", args.config+"_retrieval.json")
    with open(config_path) as fp: arg_dict = json.load(fp)

    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    device = "cpu"

    eval_args = arg_dict["eval_args"]

    # Main arguments
    dpr_path = eval_args["model"]
    task = eval_args["task"]
    batch_size = eval_args["batch_size"]
    
    dpr_model = CustomDPR.from_pretrained(model_path=dpr_path, device=device)
    #model = DRES(dpr_model, batch_size=16)
    faiss_search = FlatIPFaissSearch(dpr_model, batch_size=eval_args)

    #### Download NFCorpus dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(task)
    out_dir = "/part/01/Tmp/lvpoellhuber/datasets"
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    #print(len(corpus))

    #corpus = dict(list(corpus.items())[0:100])
    if faiss_search.faiss_index == None:
        faiss_search.index(corpus=corpus)
        faiss_search.save(dpr_path, prefix="default")


    retriever = EvaluateRetrieval(faiss_search, score_function="dot")

    print("Retrieving...")
    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    metrics_path = os.path.join(dpr_path, "metrics.txt")
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