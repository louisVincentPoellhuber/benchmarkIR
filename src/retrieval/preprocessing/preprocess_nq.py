from preprocess_utils import *

from pyserini.search.lucene import LuceneSearcher
import subprocess

def parse_arguments():
    argparser = argparse.ArgumentParser("Download NQ dataset and preprocess it.")
    argparser.add_argument('--datapath', default=STORAGE_DIR+"/datasets/nq") 
    argparser.add_argument('--overwrite', default=False) 

    args = argparser.parse_args()

    return args

def download_nq(out_dir):
    # First do Test, in case it hasn't been done yet
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
    data_path = util.download_and_unzip(url, out_dir) # We never return the test datapath, as it's already correcly formatted. 

    GenericDataLoader(data_folder=data_path).load(split="test")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq-train.zip"
    data_path = util.download_and_unzip(url, out_dir)

    return data_path

def preprocess_nq_pairs(out_dir):
    log_message(f"Pre-Processing Natural Questions.")

    data_path = download_nq(out_dir)

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


def preprocess_nq_dynamic_pairs(out_dir):
    out_dir = os.path.join(out_dir, "nq")
    log_message(f"Pre-Processing Natural Questions.")

    data_path = download_nq(out_dir)

    log_message("Loading data.")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")

    log_message("Creating train_pairs.pt file.")
    pairs = []
    for qid in tqdm(qrels.keys()):
        query = queries[qid]
        docid = list(qrels[qid].keys())[0]
        doc = corpus[docid]

        pairs.append({"query":query, "document":doc})

    dataset = PairsDataset(pairs)    
    save_path = os.path.join(out_dir,"train_pairs_dynamic.pt")
    dataset.save(save_path)
    log_message("File saved.")

def preprocess_nq_bm25(out_dir, k=6):
    data_path = download_nq(out_dir)

    log_message("Loading data.")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    
    index_dir = os.path.join(out_dir, "bm25index")
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    if len(os.listdir(index_dir)) == 0:
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
        log_message(result)
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


if __name__ == "__main__":
    
    args = parse_arguments()

    os.makedirs(args.datapath, exist_ok=True)
    
    if args.overwrite or not os.path.exists(args.datapath+"/train_pairs.pt"):
        preprocess_nq_pairs(args.datapath)
    if args.overwrite or not os.path.exists(args.datapath+"/train_pairs_dynamic.pt"):
        preprocess_nq_dynamic_pairs(args.datapath)
    if args.overwrite or not os.path.exists(args.datapath+"/train_triplets.pt"):
        preprocess_nq_bm25(args.datapath)
