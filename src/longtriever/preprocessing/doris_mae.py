from common.utils import *
from common.s2orc import *
from sci_docs import SciDocsProcessor
import time

class DorisMAEProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "doris-mae", "arxiv", overwrite)
        self.datapath = datapath
        self.subsets = ["train", "dev", "test"]

    def _extract_s2orc_texts(self):
        temp_processor = SciDocsProcessor(self.datapath, self.overwrite)
        temp_processor.download()

        db_path = os.path.join(self.corpus_dir, "corpus.db")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        ssids = set()
        for article_id in tqdm(cursor.execute("SELECT id FROM articles"), desc="Collecting all IDs from S2ORC database."):
            ssids.add(article_id[0])

        return ssids

     
    def download(self):
        file_url = "https://zenodo.org/records/8299749/files/DORIS-MAE_dataset_v1.json?download=1"

        local_path = os.path.join(self.corpus_download_dir, "DORIS-MAE_dataset_v1.json")

        if not os.path.exists(local_path) or self.overwrite:
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk) 
        else:
            log_message("JSON already exists. Skipping download.", print_message=True)
        

        with open(local_path, 'r') as file:
            corpus = json.load(file)["Corpus"]

        s2orc_ssids = self._extract_s2orc_texts()

        arxiv_ssids = set()
        for sample in corpus:
            arxiv_ssids.add(sample["arxiv_id"])

        arxiv_ssids = arxiv_ssids - s2orc_ssids

        download_url = "https://export.arxiv.org/pdf/"
        pattern = re.compile(r"^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$")

        for sample in tqdm(corpus):
            url = sample["url"]
            doi = url.split("/")[-1]
            ssid = sample["ss_id"]
            if (pattern.match(doi)) and (ssid in arxiv_ssids):
                pdf_path = os.path.join(self.corpus_download_dir, doi)+".pdf"

                if not os.path.exists(pdf_path) or self.overwrite:
                    pdf_url = download_url+doi
                    response = requests.get(pdf_url)

                    if response.status_code != 200:
                        log_message(f"Failed to download {pdf_url}. Status code: {response.status_code}", level=logging.ERROR, print_message=True)
                    else:
                        with open(pdf_path, "wb") as f:
                            f.write(response.content)

                time.sleep(0.5)
            else:
                log_message(f"Skipping invalid or already processed DOI: {doi}", level=logging.INFO, print_message=True)


    def process_corpus(self):
        corpus_path = os.path.join(self.corpus_dir, "corpus.jsonl")

        with open(corpus_path, "w", encoding="utf-8") as f:
            for pdf in os.listdir(self.corpus_download_dir):
                if pdf.endswith(".pdf"):
                    doc = fitz.open(os.path.join(dir, pdf))
                    text = ""
                    for page in doc:
                        text += page.get_text()


    def process_queries(self):
        dataset = ir_datasets.load(f"beir/hotpotqa")
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")

        if self.overwrite or not os.path.exists(queries_path):
            log_message(f"Processing queries into {queries_path}.", print_message=True)
            with open(queries_path, "w", encoding="utf-8") as f:
                for query in dataset.queries_iter():
                    obj = {
                        "_id": query.query_id,
                        "text": query.text
                    }
                    f.write(json.dumps(obj) + "\n")
        else:
            log_message(f"Queries already exist at {queries_path}. Skipping query processing.", print_message=True)

    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)
        
        db_path = os.path.join(self.corpus_download_dir, "wikipedia.db")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        for subset in self.subsets:
            dataset = ir_datasets.load(f"beir/hotpotqa/{subset}")
            qrel_path = os.path.join(self.qrel_dir, f"{subset}.tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        cursor.execute("SELECT id, title, text, url FROM articles WHERE id = ?", (qrel.doc_id,))
                        if cursor.fetchone() is not None:
                            f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)

        connection.close()

if __name__ == "__main__":
    args = parse_arguments()
    processor = DorisMAEProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
