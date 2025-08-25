from common.utils import *
from common.s2orc import *
import kagglehub

class DorisMAEProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "doris-mae", "arxiv", overwrite)
        self.datapath = datapath
        self.subsets = ["train", "dev", "test"]
        self.download_batch_size = 100

     
    def download(self):    
        file_url = "https://zenodo.org/records/8299749/files/DORIS-MAE_dataset_v1.json?download=1"
        local_path = os.path.join(self.corpus_download_dir, "DORIS-MAE_dataset_v1.json")

        # Download DORIS-MAE dataset
        if not os.path.exists(local_path) or self.overwrite:
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk) 
        
        with open(local_path, 'r') as file:
            doris_corpus = json.load(file)["Corpus"]

        # Extract Arxiv IDs from DORIS-MAE corpus
        doris_ids = set()
        doris_dict = {}
        pattern = "[+-]?([0-9]*[.])?[0-9]+"
        for sample in doris_corpus:
            id = sample["url"].split("/")[-1]
            if len(id)>2:
                if id[-2]=="v":
                    id = id[:-2]
            if re.match(pattern, id): # Filter out IDs that are not Arxiv IDs
                doris_ids.add(str(id))
                doris_dict[id] = sample["abstract_id"]

        # Download Arxiv metadatas
        path = kagglehub.dataset_download("Cornell-University/arxiv")
        dataset_path = os.path.join(path, "arxiv-metadata-oai-snapshot.json")

        # Create database for IDs and Text
        db_path = os.path.join(self.corpus_dir, "corpus.db")
        connection= sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT, 
            abstract_id INTEGER
        )
        """)
        connection.commit()

        # Extract download URLs from Arxiv papers existing in DORIS-MAE
        if len(os.listdir(self.corpus_download_dir))<=1 or self.overwrite:
            db_batch = []
            url_batch = []
            batch_nb = 0
            data = pd.read_json(dataset_path, lines=True)
            for row in tqdm(data.iterrows()):
                example = row[1]
                if example["id"] in doris_ids:
                    id = str(example["id"])
                    is_old_id = "/" in id
                    if is_old_id:
                        category = example["id"].split("/")[0]
                        id = id.split("/")[1]
                        month_id = id[:4]
                    else:
                        category = "arxiv"
                        month_id = id.split(".")[0]

                        # Zero pad
                        if len(month_id) == 3:
                            month_id = "0" + month_id

                    most_recent_version = example["versions"][-1]["version"]
                    download_url = f"gs://arxiv-dataset/arxiv/{category}/pdf/{month_id}/{id}{most_recent_version}.pdf"
                    db_batch.append((id+most_recent_version, example["title"], doris_dict[example["id"]]))
                    url_batch.append(download_url + "\n")

                if len(url_batch) >= self.download_batch_size:
                    url_path = os.path.join(self.corpus_download_dir, f"urls_{batch_nb}.txt")
                    with open(url_path, "w", encoding="utf-8") as f:
                        f.writelines(url_batch)
                    batch_nb+=1
                    url_batch.clear()

                if len(db_batch)>= 10000:
                    cursor.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?)", db_batch)
                    connection.commit()
                    db_batch.clear()
        

    def process_corpus(self):
        db_path = os.path.join(self.corpus_dir, "corpus.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database {db_path} does not exist. Please run the download step first.")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        extract_texts = True
        try:
            cursor.execute("ALTER TABLE articles ADD COLUMN text TEXT;")
        except sqlite3.OperationalError:
            log_message("Texts already processed. Skipping text extraction.", print_message=True)
            extract_texts = False


        # if extract_texts or self.overwrite:
        log_message(f"Processing corpus.", print_message=True)
        batch = []
        for file in tqdm(os.listdir(self.corpus_download_dir), desc="URL files to process"):
            # Find url_*.txt files
            if file.startswith("url"):
                # Re-make the PDF directory that's deleted every iteration
                pdf_dir = os.path.join(self.corpus_download_dir, "pdf")
                os.makedirs(pdf_dir, exist_ok=True)

                # Use gsutil to download sample of 1000 PDFs
                url_path = os.path.join(self.corpus_download_dir, file)
        
                cmd = [
                    "gsutil", "-m", "cp", "-I", pdf_dir
                ]

                # Download to pdf_dir
                with open(url_path, "r") as infile:
                    subprocess.run(cmd, stdin=infile, check=False)

                # Wait for the PDFs to finish downloading
                time.sleep(1)

                # Read the 1000 PDFs and extract their text
                for pdf in tqdm(os.listdir(pdf_dir), desc="PDFs to extract"):
                    if pdf.endswith(".pdf"):
                        text = extract_text_from_pdf(os.path.join(pdf_dir, pdf))
                        id = pdf.replace(".pdf", "")
                        batch.append((text, id))
                
                # Clear the PDF directory to save storage space
                shutil.rmtree(pdf_dir)
                    
                # Add text to DB every now and then
                cursor.executemany("UPDATE articles SET text=? WHERE id=?", batch)
                connection.commit()
                batch.clear()

        connection.close()
        # else:
        #     log_message(f"Texts already extracted. Skipping extraction.")



    def process_queries(self):
        local_path = os.path.join(self.corpus_download_dir, "DORIS-MAE_dataset_v1.json")
        with open(local_path, 'r') as file:
            dataset = json.load(file)["Corpus"]
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
    
    # processor.download()
    # processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
