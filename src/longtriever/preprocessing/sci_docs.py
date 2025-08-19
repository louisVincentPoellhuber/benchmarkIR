from common.utils import *
from common.s2orc import *
import time

class SciDocsProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "scidocs", "s2orc", overwrite)
     
    def download(self):
        ir_datasets.load("beir/scidocs")
        download_s2orc(self.corpus_download_dir, self.overwrite)
        create_db(self.corpus_download_dir, self.corpus_dir, self.overwrite)

    def process_corpus(self):
        log_message("No need to process corpus for SciDocs. Using S2ORC database directly.", print_message=True)


    def process_queries(self):
        dataset = ir_datasets.load(f"beir/scidocs")
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
        
        db_path = os.path.join(self.corpus_download_dir, "corpus.db")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        for subset in self.subsets:
            dataset = ir_datasets.load(f"beir/scidocs/{subset}")
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
    processor = SciDocsProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
