from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from dres import DenseRetrievalExactSearch as DRES
from accelerate import Accelerator, DistributedDataParallelKwargs
from beir.retrieval.models.dpr import DPR

import logging
import pathlib, os
import random
import argparse
import json

import torch


import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(STORAGE_DIR)


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
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
    argparser.add_argument('--config_dict', default={}) 
    
    args = argparser.parse_args()

    return args


if __name__ == "__main__":    
    args = parse_arguments()
    print(f"Executing {args.config} retrieval.")
   
   
    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+"_paired.json")
        with open(config_path) as fp: arg_dict = json.load(fp)

    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)


    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    device = "cpu"

    settings = arg_dict["settings"]

    # Main arguments
    dpr_path = settings["save_path"]
    batch_size = settings["batch_size"]
    
    # dpr_model = CustomDPR.from_pretrained(model_path=dpr_path, device=device)
    dpr_model = DPR(("facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-ctx_encoder-single-nq-base",))
    faiss_search = FlatIPFaissSearch(dpr_model, batch_size=batch_size)
    
    tasks = ["nq", "msmarco", "quora", "hotpotqa"]
    for task in tasks:
        print(f"\n==========================\nDoing task {task}.\n==========================\n")
        #### Download NFCorpus dataset and unzip the dataset
        # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(task)
        # out_dir = "/part/01/Tmp/lvpoellhuber/datasets" # TODO: make dynamic and update logic
        # data_path = util.download_and_unzip(url, out_dir)
        data_path = "/Tmp/lvpoellhuber/datasets/nq/nq"
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        #print(len(corpus))

        #corpus = dict(list(corpus.items())[0:100])
        if faiss_search.faiss_index == None:
            print("Indexing.")
            faiss_search.index(corpus=corpus)
            print("Saving.")
            faiss_search.save(dpr_path, prefix="default")


        retriever = EvaluateRetrieval(faiss_search, score_function="dot")

        print("Retrieving.")
        results = retriever.retrieve(corpus, queries)

        print("Evaluating.")
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