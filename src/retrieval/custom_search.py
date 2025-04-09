from beir.retrieval.search.dense import FlatIPFaissSearch
from beir.retrieval.search.dense.faiss_index import FaissIndex
import logging
import faiss
import numpy as np
import os
logger = logging.getLogger(__name__)

class CustomFaissSearch(FlatIPFaissSearch):
    def __init__(self, model, index_path, batch_size = 128, corpus_chunk_size = 50000, use_gpu = False, **kwargs):
        super().__init__(model, batch_size, corpus_chunk_size, use_gpu, **kwargs)
        self.index_path = index_path

    def index(self, corpus: dict[str, dict[str, str]], score_function: str = None, **kwargs):
        if os.path.exists(os.path.join(self.index_path, "corpus_embeddings.npy")) and os.path.exists(os.path.join(self.index_path, "faiss_ids.npy")):
            logger.info("Loading existing index...")
            corpus_embeddings = np.load(os.path.join(self.index_path, "corpus_embeddings.npy"))
            faiss_ids = np.load(os.path.join(self.index_path, "faiss_ids.npy"))
        else:
            faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)
            logger.info("Saving temporary index...")
            np.save(os.path.join(self.index_path, "corpus_embeddings.npy"), corpus_embeddings)
            np.save(os.path.join(self.index_path, "faiss_ids.npy"), faiss_ids)

        logger.info("Initializing Index...")
        base_index = faiss.IndexFlatIP(self.dim_size)
        if self.use_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            logger.info("Building Faiss Index...")
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, base_index)
