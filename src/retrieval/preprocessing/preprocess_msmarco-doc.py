from preprocess_utils import *
import gzip
import csv
import requests
import pandas as pd
from random import randint
from pyserini.search.lucene import LuceneSearcher
import subprocess
import re

def parse_arguments():
    argparser = argparse.ArgumentParser("Download MSMARCO dataset and preprocess it.")
    argparser.add_argument('--datapath', default=STORAGE_DIR+"/datasets/msmarco-doc") 
    argparser.add_argument('--overwrite', default=False) 

    args = argparser.parse_args()

    return args

def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with (
        open(save_path, "wb") as fd,
        tqdm(
            desc=save_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar,
    ):
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

# This function seeks the docid in the TSV file. It's to make it easier to access since it's so huge. 
def getcontent(docoffset, docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    return line.rstrip().split("\t")


def download_msmarco(out_dir):
    files = ["msmarco-docs.tsv.gz", "msmarco-docs-lookup.tsv.gz", "msmarco-doctrain-queries.tsv.gz", "msmarco-doctrain-top100.gz", "msmarco-doctrain-qrels.tsv.gz", "msmarco-docdev-queries.tsv.gz", "msmarco-docdev-top100.gz", "msmarco-docdev-qrels.tsv.gz", "docleaderboard-queries.tsv.gz", "docleaderboard-top100.gz"]
    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"

    filenames = "\t\n| ".join(files)
    log_message(f"Downloading files: {filenames}")
    for download_file in files:
        file_url = url + download_file
        save_path = os.path.join(out_dir, download_file)
        
        if os.path.exists(save_path):
            log_message(f"File {download_file} already exists, skipping.")
        else:
            download_url(url = file_url, save_path = save_path)


def preprocess_corpus(out_dir):
    corpus_filepah = os.path.join(out_dir, "corpus.jsonl")

    log_message("Writing corpus to disk")
    with open(corpus_filepah, "w") as corpus_out, \
        gzip.open(os.path.join(out_dir, "msmarco-docs.tsv.gz"), 'rt', encoding='utf8') as corpus_in:
        print("Reading corpus all at once. This will take about two minutes.")
        corpus_in_content = corpus_in.readlines()
        
        skipped_docs = ["D2070532", "D705499", "D3246147", "D3414215", "D3336081", "D3259427"] # Known problematic documentss
        for doc_line in tqdm(corpus_in_content):
            doc = doc_line.rstrip().split("\t")
            if len(doc) == 4:
                json.dump({"_id": doc[0], "title": doc[2], "text": doc[3]}, corpus_out)
                corpus_out.write("\n")
            else:
                skipped_docs.append(doc[0])

    skipped_doc_path = os.path.join(out_dir, "skipped_docs.csv")
    pd.Series(skipped_docs).to_csv(skipped_doc_path, index=False, header=None)

def preprocess_queries(out_dir):
    querystring = {}
    with gzip.open(os.path.join(out_dir, "msmarco-doctrain-queries.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            querystring[topicid] = querystring_of_topicid
    with gzip.open(os.path.join(out_dir, "msmarco-docdev-queries.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            querystring[topicid] = querystring_of_topicid
    

    log_message("Writing queries to disk")
    queries_filepah = os.path.join(out_dir, "queries.jsonl")

    with open(queries_filepah, "w") as f:
        for qid, query in tqdm(querystring.items()):
            json.dump({"_id": qid, "text": query, "metadata": {}}, f)
            f.write("\n")

def preprocess_dev(out_dir):
    skipped_doc_path = os.path.join(out_dir, "skipped_docs.csv")
    skipped_docs = pd.read_csv(skipped_doc_path, header=None)[0].to_list()
    
    dev_qrel = {}
    with gzip.open(os.path.join(out_dir, "msmarco-docdev-qrels.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for item in tsvreader:
            topicid, _, docid, rel = item[0].split(" ")
            assert rel == "1"
            if docid not in skipped_docs: # Skip  documents without text
                dev_qrel[topicid] = docid

    log_message("Writing qrels to disk")
    qrel_filepath = os.path.join(qrel_dir, "test.tsv")

    with open(qrel_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator='\n')
        
        writer.writerow(["query-id", "corpus-id", "score"])
        
        for qid, docid in dev_qrel.items():
            writer.writerow([qid, docid, 1])


def preprocess_train(out_dir):
    skipped_doc_path = os.path.join(out_dir, "skipped_docs.csv")
    skipped_docs = pd.read_csv(skipped_doc_path, header=None)[0].to_list()
    
    train_qrel = {}
    with gzip.open(os.path.join(out_dir, "msmarco-doctrain-qrels.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for item in tsvreader:
            topicid, _, docid, rel = item[0].split(" ")
            assert rel == "1"
            if docid not in skipped_docs: # Skip  documents without text
                train_qrel[topicid] = docid

    log_message("Writing qrels to disk")
    qrel_filepath = os.path.join(qrel_dir, "train.tsv")

    with open(qrel_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator='\n')
        
        writer.writerow(["query-id", "corpus-id", "score"])
        
        for qid, docid in train_qrel.items():
            writer.writerow([qid, docid, 1])


def preprocess_train_pairs(out_dir):
    skipped_doc_path = os.path.join(out_dir, "skipped_docs.csv")
    skipped_docs = pd.read_csv(skipped_doc_path, header=None)[0].to_list()

    log_message("Loading data.")
    corpus_filepah = os.path.join(out_dir, "corpus.jsonl")
    corpus = load_jsonl(corpus_filepah)  # Use the new function to load the JSONL file
    
    train_querystring = {}
    with gzip.open(os.path.join(out_dir, "msmarco-doctrain-queries.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            train_querystring[topicid] = querystring_of_topicid
    
    train_qrel = {}
    with gzip.open(os.path.join(out_dir, "msmarco-doctrain-qrels.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for item in tsvreader:
            topicid, _, docid, rel = item[0].split(" ")
            assert rel == "1"
            if docid not in skipped_docs: # Skip documents without text
                train_qrel[topicid] = docid


    log_message("Creating train_pairs.pt file.")
    pairs_queries = []
    pairs_docs = []
    for qid in tqdm(train_qrel.keys()):
        query = train_querystring[qid]
        docid = train_qrel[qid]
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


def preprocess_short(data_path, reduction_factor=0.08):
    out_dir = data_path+"-short"
    os.makedirs(out_dir, exist_ok=True)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    nb_queries = round(reduction_factor * len(queries))
    nb_documents = round(reduction_factor * len(corpus))

    query_ids = list(queries.keys())

    short_corpus = {}
    short_queries = {}
    short_qrels = {}

    randid = randint(0, len(queries))
    qid = query_ids[randid]
    for i in tqdm(range(nb_queries)):
        while qid in short_queries.keys():
            randid = randint(0, len(queries))
            qid = query_ids[randid]

        query = queries[qid]
        docid = list(qrels[qid].keys())[0]
        short_corpus[docid] = corpus[docid]
        
        short_qrels[qid] = docid
        short_queries[qid] = query

    positive_docids = short_corpus.keys()
    all_docids = list(corpus.keys())

    while (len(short_corpus) < nb_documents):
        random_int = randint(0, len(corpus))
        random_docid = all_docids[random_int]

        if random_docid not in positive_docids:
            short_corpus[random_docid] = corpus[random_docid]

    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, doc_content in short_corpus.items():
            json.dump({"_id": doc_id, **doc_content}, f)
            f.write("\n") 

    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for qid, query in short_queries.items():
            json.dump({"_id": qid, "text": query, "metadata": {}}, f)
            f.write("\n") 
    
    # Save as TSV file
    qrel_dir = os.path.join(out_dir, "qrels")
    os.makedirs(qrel_dir, exist_ok=True)
    with open(os.path.join(qrel_dir, "test.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")  # Use tab as delimiter
        
        # Write header
        writer.writerow(["query-id", "corpus-id", "score"])
        
        # Write rows
        for qid, docid in short_qrels.items():
            writer.writerow([qid, docid, 1])

    dataloader = get_pairs_dataloader(batch_size = 1, dataset_path=os.path.join(data_path, "train_pairs.pt"))
    
    pairs_queries =[]
    pairs_docs =[]
    loop = tqdm(dataloader)
    max_nb_pairs = round(reduction_factor * len(dataloader))
    for i, batch in enumerate(loop):
        if i>=max_nb_pairs:
            break
        query = batch["queries"][0]
        document = batch["documents"][0]

        pairs_queries.append(query)
        pairs_docs.append(document)

    pairs = {
        "queries":pairs_queries,
        "documents":pairs_docs
    }

    dataset = PairsDataset(pairs)    
    save_path = os.path.join(out_dir,"train_pairs.pt")
    dataset.save(save_path)

def index_corpus(out_dir):
    index_dir = os.path.join(out_dir, "bm25index")
    os.makedirs(index_dir, exist_ok=True)
    corpus_filepah = os.path.join(index_dir, "corpus.jsonl")

    if len(os.listdir(index_dir)) <= 1 :
        log_message("Writing corpus to disk")
        with open(corpus_filepah, "w") as corpus_out, \
            gzip.open(os.path.join(out_dir, "msmarco-docs.tsv.gz"), 'rt', encoding='utf8') as corpus_in:
            print("Reading corpus all at once. This will take about two minutes.")
            corpus_in_content = corpus_in.readlines()
            
            skipped_docs = ["D2070532", "D705499", "D3246147", "D3414215", "D3336081", "D3259427"] # Known problematic documentss
            for doc_line in tqdm(corpus_in_content):
                doc = doc_line.rstrip().split("\t")
                if len(doc) == 4:
                    json.dump({"id": doc[0], "title": doc[2], "contents": doc[3]}, corpus_out)
                    corpus_out.write("\n")
                else:
                    skipped_docs.append(doc[0])

        log_message("Executing Pyserini indexing command. Note: Errors might appear, but indexing should run properly regardless.")
        cmd = [
            "python", "-m", "pyserini.index",
            "-collection", "JsonCollection",
            "-input", index_dir,
            "-index", index_dir,
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", "4",
            "-storePositions", "-storeDocvectors", "-storeRaw"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Print each line as it's received
        for line in process.stdout:
            print(line, end='')  # `end=''` avoids double newlines

        # Wait for process to finish
        process.wait()
    else:
        log_message("Corpus already indexed.")

    searcher = LuceneSearcher(index_dir)

    return searcher
    
def preprocess_nqrels(out_dir):
    searcher = index_corpus(out_dir)
    nqrel_dir = os.path.join(out_dir, "nqrels")
    os.makedirs(nqrel_dir, exist_ok=True)
    corpus, queries, qrels = GenericDataLoader(data_folder=out_dir).load(split="train")
    

    with open(os.path.join(nqrel_dir, "train.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")  # Use tab as delimiter
        
        # Write header
        writer.writerow(["query-id", "corpus-id", "score"])

        for qid in tqdm(qrels.keys()):
            # Add query
            query=queries[qid]
            subbed_query = re.sub(r'[^\w\s]',' ',query)
            
            # Add positive document
            pos_docid = list(qrels[qid].keys())[0]

            # Get the negative documents using BM25
            # NOTE: There is only one negative document. If you want more, uncomment the code below. 
            negatives = searcher.search(subbed_query, k=2)
            if len(negatives) == 0:
                log_message(f"No negative documents found for query {qid}.")

            negative_pid = negatives[0].docid if negatives[0].docid != pos_docid else negatives[1].docid
            # pos_doc = corpus[pos_docid]
            # neg_doc = corpus[negative_pid]
            
            writer.writerow([qid, negative_pid, -1])

            
def preprocess_bm25(out_dir, k=2):
    searcher = index_corpus(out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=out_dir).load(split="train")
    
    triplet_queries = []
    triple_pos_docs = []
    triple_neg_docs = []
    for qid in tqdm(qrels.keys()):
        # Add query
        query = queries[qid]
        triplet_queries.append(query)
        
        # Add positive document
        pos_docid = list(qrels[qid].keys())[0]
        pos_doc = corpus[pos_docid]
        triple_pos_docs.append(pos_doc)

        # Get the negative documents using BM25
        # NOTE: There is only one negative document. If you want more, uncomment the code below. 
        negatives = searcher.search(query, k=k)
        # neg_docs = []
        # for i in range(len(negatives)):
        #     if (negatives[i].docid != pos_docid) & (len(neg_docs)<k-1):
        #         neg_docs.append(corpus[negatives[i].docid])
        # triple_neg_docs.append(neg_docs)
        if len(negatives) > 0:
            negative_pid = negatives[0].docid if negatives[0].docid != pos_docid else negatives[1].docid
            negative_doc = corpus[negative_pid]
        triple_neg_docs.append(negative_doc)

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
    out_dir = args.datapath

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    qrel_dir = os.path.join(out_dir, "qrels")
    os.makedirs(qrel_dir, exist_ok=True)

    index_dir = os.path.join(out_dir, "bm25index")
    os.makedirs(index_dir, exist_ok=True)

    download_msmarco(out_dir)

    if args.overwrite or not os.path.exists(out_dir+"/corpus.jsonl"):
        preprocess_corpus(out_dir)
    if args.overwrite or not os.path.exists(out_dir+"/queries.jsonl"):
        preprocess_queries(out_dir)
    if args.overwrite or not os.path.exists(qrel_dir+"/test.tsv"):
        preprocess_dev(out_dir)
    if args.overwrite or not os.path.exists(qrel_dir+"/train.tsv"):
        preprocess_train(out_dir)
    if args.overwrite or not os.path.exists(out_dir+"/train_pairs.pt"):
        preprocess_train_pairs(out_dir)
    if args.overwrite or not os.path.exists(out_dir+"-short"):
        preprocess_short(out_dir)
    if args.overwrite or not os.path.exists(index_dir+"/corpus.jsonl") or not os.path.exists(out_dir+"/train_triplets.pt"):
        preprocess_bm25(out_dir)
    if args.overwrite or not os.path.exists(index_dir+"/corpus.jsonl") or not os.path.exists(out_dir+"/nqrels/train.tsv"):
        preprocess_nqrels(out_dir)
        