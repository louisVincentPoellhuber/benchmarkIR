from preprocess_utils import *
import gzip
import csv
import requests
import pandas as pd

def parse_arguments():
    argparser = argparse.ArgumentParser("Download MSMARCO dataset and preprocess it.")
    argparser.add_argument('--tasks', default=["all"], help="Which tasks to preprocess. Options: nq, nq_dynamic, nq_bm25, all") # wikipedia
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


def preprocess_dev(out_dir):
    skipped_doc_path = os.path.join(out_dir, "skipped_docs.csv")
    skipped_docs = pd.read_csv(skipped_doc_path, header=None)[0].to_list()
    
    dev_querystring = {}
    with gzip.open(os.path.join(out_dir, "msmarco-docdev-queries.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            dev_querystring[topicid] = querystring_of_topicid

    dev_qrel = {}
    with gzip.open(os.path.join(out_dir, "msmarco-docdev-qrels.tsv.gz"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for item in tsvreader:
            topicid, _, docid, rel = item[0].split(" ")
            assert rel == "1"
            if topicid not in skipped_docs: # Skip  documents without text
                dev_qrel[topicid] = docid

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
        for qid, query in tqdm(dev_querystring.items()):
            json.dump({"_id": qid, "text": query, "metadata": {}}, f)
            f.write("\n")

def load_jsonl(filepath):
    corpus = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())  # Parse each line as a JSON object
            corpus[doc["_id"]] = doc  # Use the document ID as the key
    return corpus

def preprocess_train(out_dir):
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

def preprocess_corpus(out_dir):
    corpus_filepah = os.path.join(out_dir, "corpus.jsonl")

    log_message("Writing corpus to disk")
    with open(corpus_filepah, "w") as corpus_out, \
        gzip.open(os.path.join(out_dir, "msmarco-docs.tsv.gz"), 'rt', encoding='utf8') as corpus_in:
        print("Reading corpus all at once. This will take about two minutes.")
        corpus_in_content = corpus_in.readlines()
        
        skipped_docs = []
        for doc_line in tqdm(corpus_in_content):
            doc = doc_line.rstrip().split("\t")
            if len(doc) == 4:
                json.dump({"_id": doc[0], "title": doc[2], "text": doc[3]}, corpus_out)
                corpus_out.write("\n")
            else:
                skipped_docs.append(doc[0])

    skipped_doc_path = os.path.join(out_dir, "skipped_docs.csv")
    pd.Series(skipped_docs).to_csv(skipped_doc_path, index=False, header=None)

if __name__ == "__main__":
    
    args = parse_arguments()
    out_dir = args.datapath

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    download_msmarco(out_dir)

    if args.overwrite or not os.path.exists(args.datapath+"/corpus.jsonl"):
        preprocess_corpus(out_dir)
    if args.overwrite or not os.path.exists(args.datapath+"/queries.jsonl"):
        preprocess_dev(out_dir)
    if args.overwrite or not os.path.exists(args.datapath+"/train_pairs.pt"):
        preprocess_train(out_dir)

    