from common.utils import *
from common.wikipedia import *

class NQProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "nq", "wikipedia", overwrite)
     
    def download(self):
        ir_datasets.load("natural-questions")
        download_wikipedia(self.corpus_dir, self.overwrite)

    def process_corpus(self):
        corpus = ir_datasets.load("natural-questions")

        doc_titles = []
        for doc in tqdm(corpus.docs_iter(), total=corpus.docs_count()):
            if doc.doc_id.split("-")[1]=="0": # We don't care about all the sub-passages, only the first one
                doc_titles.append(doc.document_title)   

        # process_pages(doc_titles, self.corpus_download_dir)

    def process_queries(self):
        pass

    def process_qrels(self):
        pass


if __name__ == "__main__":
    args = parse_arguments()
    processor = NQProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
    processor.process_train_pairs()