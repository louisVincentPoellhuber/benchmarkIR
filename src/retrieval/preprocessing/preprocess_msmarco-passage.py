from preprocess_utils import *
import gzip
import csv
import requests
import pandas as pd
from random import randint
import tarfile
import subprocess
from pyserini.search.lucene import LuceneSearcher
from huggingface_hub import hf_hub_download

def parse_arguments():
    argparser = argparse.ArgumentParser("Download MSMARCO dataset and preprocess it.")
    argparser.add_argument('--datapath', default=STORAGE_DIR+"/datasets/msmarco-passage") 
    argparser.add_argument('--overwrite', default=True) 

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
    files = ["collection.tar.gz", "queries.tar.gz", "qrels.dev.tsv", "qrels.train.tsv", "qidpidtriples.train.full.2.tsv.gz"]
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
        tarfile.open(os.path.join(out_dir, "collection.tar.gz"), 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f is not None:
                    print("Reading corpus all at once.")
                    corpus_in_content = f.readlines()
                    
                    for i, doc_line in enumerate(tqdm(corpus_in_content)):
                        doc = doc_line.decode("utf-8").rstrip().split("\t")
                        json.dump({"_id": doc[0],"text": doc[1]}, corpus_out)
                        corpus_out.write("\n")

def preprocess_queries(out_dir):
    
    querystrings = {}
    with tarfile.open(os.path.join(out_dir, "queries.tar.gz"), 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f is not None:
                    queries_in_content = f.readlines()
                    for i, line in enumerate(tqdm(queries_in_content)):
                        query = line.decode("utf-8").rstrip().split("\t")
                        qid = query[0]
                        querystr = query[1]
                        querystrings[qid] = querystr
    
    log_message("Writing queries to disk")
    queries_filepah = os.path.join(out_dir, "queries.jsonl")

    with open(queries_filepah, "w") as f:
        for qid, query in tqdm(querystrings.items()):
            json.dump({"_id": qid, "text": query, "metadata": {}}, f)
            f.write("\n")

def preprocess_dev(out_dir):
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


def preprocess_train(out_dir):

    train_qrel = {}
    with open(os.path.join(out_dir, "qrels.train.tsv"), 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for item in tsvreader:
            qid, _, docid, rel = item
            assert rel == "1"
            train_qrel[qid] = docid

    log_message("Writing qrels to disk")
    qrel_dir = os.path.join(out_dir, "qrels")
    if not os.path.exists(qrel_dir):
        os.makedirs(qrel_dir)
    qrel_filepath = os.path.join(qrel_dir, "train.tsv")

    with open(qrel_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator='\n')
        
        writer.writerow(["query-id", "corpus-id", "score"])
        
        for qid, docid in train_qrel.items():
            writer.writerow([qid, docid, 1])
        

def preprocess_train_pairs(out_dir):
    log_message("Loading data.")
    corpus, queries, qrels = GenericDataLoader(data_folder=out_dir).load(split="train")

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
        if qid in queries.keys():
            query = queries[qid]
            docid = train_qrel[qid]
            doc = corpus[docid]
            if "title" in doc:
                doc.pop("title")

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


def preprocess_nqrels(out_dir):
    nqrel_dir = os.path.join(out_dir, "nqrels")
    os.makedirs(nqrel_dir, exist_ok=True)
    train_qrels_path = os.path.join(out_dir, "qrels", "train.tsv")
    triples_out_path = os.path.join(nqrel_dir, "train.tsv")

    hard_negative_path = hf_hub_download(repo_id="sentence-transformers/msmarco-hard-negatives", filename="msmarco-hard-negatives.jsonl.gz", repo_type="dataset", cache_dir=STORAGE_DIR+"/datasets/msmarco-passage")

    with open(triples_out_path, "w") as triples_out, \
    gzip.open(hard_negative_path, 'rt', encoding='utf8') as hard_negatives, \
    open(train_qrels_path, 'rt', encoding='utf8') as train_qrels_in:

        writer = csv.writer(triples_out, delimiter="\t") 
        writer.writerow(["query-id", "corpus-id", "score"])

        negatives = {}
        for line in tqdm(hard_negatives):
            line = json.loads(line.rstrip())
            line_negatives = line["neg"]
            if "bm25" in line_negatives.keys():
                negatives[line["qid"]] =  line_negatives["bm25"][0]
            else:
                negatives[line["qid"]] =  line_negatives[list(line_negatives.keys())[0]][0]


        # The NQRELS need to have the same order as the train qrels, so we read them in order.
        qrels_content = train_qrels_in.readlines()[1:]
        for line in tqdm(qrels_content):
            qid, docid, score = line.rstrip().split("\t")
            qid=int(qid)
            if qid not in negatives.keys():
                print(f"{qid} not in NQRELS. ")
            
            nid = negatives[qid]
            writer.writerow([qid, nid, -1])
            


def preprocess_hard_negatives(out_dir, k=2):
    
    index_dir = os.path.join(out_dir, "bm25index")
    os.makedirs(index_dir, exist_ok=True)
    corpus_filepah = os.path.join(index_dir, "corpus.jsonl")

    if not os.path.exists(corpus_filepah):        
        log_message("Writing corpus to disk")
        with open(corpus_filepah, "w") as corpus_out, \
        tarfile.open(os.path.join(out_dir, "collection.tar.gz"), 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        print("Reading corpus all at once.")
                        corpus_in_content = f.readlines()
                        
                        for i, doc_line in enumerate(tqdm(corpus_in_content)):
                            doc = doc_line.decode("utf-8").rstrip().split("\t")
                            json.dump({"id": doc[0],"contents": doc[1]}, corpus_out)
                            corpus_out.write("\n")

    if len(os.listdir(index_dir)) <= 5:
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
        if "title" in pos_doc:
            pos_doc.pop("title")
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
            if "title" in negative_doc:
                negative_doc.pop("title")
        triple_neg_docs.append(negative_doc) # We use the previous negative document if none are returned... Not ideal but better than nothing. 

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

    os.makedirs(out_dir, exist_ok=True)

    qrel_dir = os.path.join(out_dir, "qrels")
    os.makedirs(qrel_dir, exist_ok=True)

    download_msmarco(out_dir)


    # if args.overwrite or not os.path.exists(out_dir+"/corpus.jsonl"):
    #     preprocess_corpus(out_dir)
    # if args.overwrite or not os.path.exists(out_dir+"/queries.jsonl"):
    #     preprocess_queries(out_dir)
    # if args.overwrite or not os.path.exists(qrel_dir+"/test.tsv"):
    #     preprocess_dev(out_dir)
    # if args.overwrite or not os.path.exists(qrel_dir+"/train.tsv"):
    #     preprocess_train(out_dir)
    # if args.overwrite or not os.path.exists(out_dir+"/train_pairs.pt"):
    #     preprocess_train_pairs(out_dir)
    # if args.overwrite or not os.path.exists(out_dir+"/train_triplets.pt"):
    #     preprocess_hard_negatives(out_dir)
    if args.overwrite or not os.path.exists(out_dir+"/nqrels/train.tsv"):
        preprocess_nqrels(out_dir)

    