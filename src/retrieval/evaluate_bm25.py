"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch.

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""

import logging
import os
import pathlib
import random

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

logging.basicConfig( 
    encoding="utf-8", 
    filename=f"bm25.log", 
    filemode="a", 
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
    )

def log_message(message, level=logging.WARNING):
    print(message)
    logging.log(msg=message, level=level)
    
class LuceneBM25(BM25):
    def __init__(self, index_path, *args, **kwargs):
        super().__init__(index_name=index_path.split("/")[-1], *args, **kwargs)
        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)

    def search(self, corpus, queries, top_k, score_function=None):
        for qid in tqdm(qrels.keys()):
            scores = {}
            query = queries[qid]
            
            results = self.searcher.search(query, k=top_k)
            if len(results) > 0:
                for i in range(len(results)):
                    doc_id = results[i].docid
                    score = results[i].score
                    scores[doc_id] = score
            else:
                raise ValueError(f"No results found for query {qid}")
            
            self.results[qid] = scores
        return self.results


    def index(self, corpus):
        print("Indexing not implemented! Do it yourself bestie.")

log_message(f"========================= Evaluating run BM25.=========================")

corpus, queries, qrels = GenericDataLoader("/Tmp/lvpoellhuber/datasets/msmarco-doc").load(split="test")

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

index_path = "/Tmp/lvpoellhuber/datasets/msmarco-doc/bm25index" 

#### Intialize ####
# (1) True - Delete existing index and re-index all documents from scratch
# (2) False - Load existing index
initialize = True  # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
# SciFact is a relatively small dataset! (limit shards to 1)
number_of_shards = 1
model = LuceneBM25(index_path=index_path)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
log_message(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(results.items()))
log_message(f"Query : {queries[query_id]}\n")

with open("bm25_metrics.txt", "w") as metrics_file:
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

