from beir.retrieval.search.dense import FlatIPFaissSearch
import importlib.util
import logging
import os

if importlib.util.find_spec("faiss") is not None:
    import faiss

from beir.retrieval.search.dense.faiss_index import FaissBinaryIndex, FaissHNSWIndex, FaissIndex, FaissTrainIndex
logger = logging.getLogger(__name__)


class StreamedFlatIPFaissSearch(FlatIPFaissSearch):
    def index(self, corpus: dict[str, dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)
        base_index = faiss.IndexFlatIP(self.dim_size)
        if self.use_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "flat"):
        super().save(output_dir, prefix, ext)

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        score_function=str,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        return super().search(corpus, queries, top_k, score_function, **kwargs)
