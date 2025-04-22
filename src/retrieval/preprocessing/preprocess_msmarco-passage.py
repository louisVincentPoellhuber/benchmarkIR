from preprocess_utils import *
import gzip
import csv
import requests
import pandas as pd
from random import randint

def parse_arguments():
    argparser = argparse.ArgumentParser("Download MSMARCO dataset and preprocess it.")
    argparser.add_argument('--datapath', default=STORAGE_DIR+"/datasets/msmarco-passage") 
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


def download_msmarco(out_dir): # NOTE: I can also download the triples directly. 
    files = ["collection.tar.gz", "queries.tar.gz", "qrels.dev.tsv", "qrels.train.tsv"]
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


def preprocess_dev(out_dir):
    querystrings = {}
    with gzip.open(os.path.join(out_dir, "queries.tar.gz"), 'rt', encoding='utf8') as f:
        queries_in_content = f.readlines()
        for i, line in enumerate(tqdm(queries_in_content)):
            query = line.rstrip().split("\t")
            if (i==0) & (len(query)==2):    
                # qid=query[0].strip("queries.dev.tsv\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000664\x000001750\x000001750\x0000020762502\x0013372646016\x00014016\x00 0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00ustar  \x00erasmus\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00erasmus\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
                qid="1048578"
                querystr = query[1]
                querystrings[qid] = querystr
            elif len(query) == 2:
                qid = query[0]
                querystr = query[1]
                querystrings[qid] = querystr

    dev_qrel = {}
    with open(os.path.join(out_dir, "qrels.dev.tsv"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for item in tsvreader:
            qid, _, docid, rel = item
            assert rel == "1"
            dev_qrel[qid] = docid

    log_message("Writing qrels to disk")
    qrel_dir = os.path.join(out_dir, "qrels")
    if not os.path.exists(qrel_dir):
        os.makedirs(qrel_dir)
    qrel_filepath = os.path.join(qrel_dir, "test.tsv")


    with open(qrel_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator='\n')
        
        writer.writerow(["query-id", "corpus-id", "score"])
        
        for qid, docid in dev_qrel.items():
            writer.writerow([qid, docid, 1])
        
    log_message("Writing queries to disk")
    queries_filepah = os.path.join(out_dir, "queries.jsonl")

    with open(queries_filepah, "w") as f:
        for qid, query in tqdm(querystrings.items()):
            json.dump({"_id": qid, "text": query, "metadata": {}}, f)
            f.write("\n")

def load_jsonl(filepath):
    corpus = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())  # Parse each line as a JSON object
            corpus[doc["_id"]] = doc  # Use the document ID as the key
    return corpus

def preprocess_train_pairs(out_dir):
    log_message("Loading data.")
    corpus_filepah = os.path.join(out_dir, "corpus.jsonl")
    corpus = load_jsonl(corpus_filepah)  # Use the new function to load the JSONL file
    
    querystrings = {}
    with gzip.open(os.path.join(out_dir, "queries.tar.gz"), 'rt', encoding='utf8') as f:
        queries_in_content = f.readlines()
        for i, line in enumerate(tqdm(queries_in_content)):
            query = line.rstrip().split("\t")
            if (i==0) & (len(query)==2):    
                # qid=query[0].strip("queries.dev.tsv\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000664\x000001750\x000001750\x0000020762502\x0013372646016\x00014016\x00 0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00ustar  \x00erasmus\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00erasmus\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
                qid="1048578"
                querystr = query[1]
                querystrings[qid] = querystr
            elif len(query) == 2:
                qid = query[0]
                querystr = query[1]
                querystrings[qid] = querystr

    train_qrel = {}
    with open(os.path.join(out_dir, "qrels.train.tsv"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for item in tsvreader:
            qid, _, docid, rel = item
            assert rel == "1"
            train_qrel[qid] = docid


    log_message("Creating train_pairs.pt file.")
    pairs_queries = []
    pairs_docs = []
    for qid in tqdm(train_qrel.keys()):
        if qid in querystrings.keys():
            query = querystrings[qid]
            docid = train_qrel[qid]
            doc = corpus[docid]

            pairs_queries.append(query)
            pairs_docs.append(doc)
        else:
            print(f"Query {qid} not found in queries.")

    pairs = {
        "queries":pairs_queries,
        "documents":pairs_docs
    }

    dataset = PairsDataset(pairs)    
    save_path = os.path.join(out_dir,"train_pairs.pt")
    dataset.save(save_path)
    log_message("File saved.")

def preprocess_corpus(out_dir):
    corpus_filepah = os.path.join(out_dir, "corpus.jsonl")

    log_message("Writing corpus to disk")
    with open(corpus_filepah, "w") as corpus_out, \
        gzip.open(os.path.join(out_dir, "collection.tar.gz"), 'rt', encoding='utf8') as corpus_in:
        print("Reading corpus all at once.")
        corpus_in_content = corpus_in.readlines()
        
        for i, doc_line in enumerate(tqdm(corpus_in_content)):
            doc = doc_line.rstrip().split("\t")
            if (len(doc)==2) & (i==0):
                json.dump({"_id": '0',"text": doc[1]}, corpus_out)
                corpus_out.write("\n")
            elif len(doc) == 2:
                json.dump({"_id": doc[0],"text": doc[1]}, corpus_out)
                corpus_out.write("\n")

if __name__ == "__main__":
    
    args = parse_arguments()
    out_dir = args.datapath

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    download_msmarco(out_dir)

    if args.overwrite or not os.path.exists(out_dir+"/corpus.jsonl"):
        preprocess_corpus(out_dir)
    if args.overwrite or not os.path.exists(out_dir+"/queries.jsonl"):
        preprocess_dev(out_dir)
    if args.overwrite or not os.path.exists(out_dir+"/train_pairs.pt"):
        preprocess_train_pairs(out_dir)

    