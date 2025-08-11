from common.utils import *
from common.wikipedia import *

class HotPotQAProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "hotpotqa", "wikipedia", overwrite)
        self.subsets = ["train", "dev", "test"]
     
    def download(self):
        ir_datasets.load("beir/hotpotqa")

        if len(os.listdir(self.corpus_download_dir))>1:
            create_db(self.corpus_download_dir, self.corpus_dir, self.overwrite)
        else:
            raise Exception("No extracted Wikipedia folders found. Please run 'process_wikipedia.sh' first.")

    def process_corpus(self):
        # corpus = ir_datasets.load("beir/hotpotqa")

        # doc_titles = []
        # for doc in tqdm(corpus.docs_iter(), total=corpus.docs_count()):
        #     doc_titles.append(doc.title)   
        pass
            

    def process_queries(self):
        dataset = ir_datasets.load(f"beir/hotpotqa")
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")

        if self.overwrite or not os.path.exists(queries_path):
            with open(queries_path, "w", encoding="utf-8") as f:
                for query in dataset.queries_iter():
                    obj = {
                        "_id": query.query_id,
                        "text": query.text
                    }
                    f.write(json.dumps(obj) + "\n")
        else:
            print(f"Queries already exist. Skipping query processing.")

    def process_qrels(self):
        for subset in self.subsets:
            dataset = ir_datasets.load(f"beir/hotpotqa/{subset}")
            qrel_path = os.path.join(self.qrel_dir, f"{subset}.tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            else:
                print(f"Qrels for {subset} already exist. Skipping qrel processing.")

if __name__ == "__main__":
    args = parse_arguments()
    processor = HotPotQAProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
    processor.process_train_pairs()