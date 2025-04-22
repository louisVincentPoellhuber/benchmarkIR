import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.nn.functional as F
from beir.retrieval.models.util import extract_corpus_sentences
from tqdm import trange
from data_handler import DataCollatorForEvaluatingLongtriever

class BertRetriever(nn.Module):
    def __init__(self,model, batch_size=6, tokenizer_path="bert-base-uncased"):
        super().__init__()
        self.encoder=model
        self.batch_size=batch_size
        self.sep=" [SEP] "
        self.data_collator = DataCollatorForEvaluatingLongtriever(tokenizer_path, 512, 512, 8)

    def save_pretrained(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[dist.get_rank()] = t
        return all_tensors

    def forward(self, query_input_ids, query_attention_mask, corpus_input_ids, corpus_attention_mask):
        query_embeddings=self.encoder(query_input_ids,query_attention_mask,return_dict=True).last_hidden_state[:, 0]
        corpus_embeddings=self.encoder(corpus_input_ids,corpus_attention_mask,return_dict=True).last_hidden_state[:, 0]
        co_query_embeddings = torch.cat(self._gather_tensor(query_embeddings.contiguous()))
        co_corpus_embeddings = torch.cat(self._gather_tensor(corpus_embeddings.contiguous()))
        co_loss = self.compute_contrastive_loss(co_query_embeddings, co_corpus_embeddings)
        return (co_loss,)

    def compute_contrastive_loss(self, co_query_embeddings, co_corpus_embeddings):
        similarities_1 = torch.matmul(co_query_embeddings, co_corpus_embeddings.transpose(0, 1))
        similarities_2 = torch.matmul(co_query_embeddings, co_query_embeddings.transpose(0, 1))
        similarities_2.fill_diagonal_(float('-inf'))
        similarities=torch.cat([similarities_1,similarities_2],dim=1)
        labels=torch.arange(similarities.shape[0],dtype=torch.long,device=similarities.device)
        co_loss = F.cross_entropy(similarities, labels) * dist.get_world_size()
        return co_loss
    
    def encode_corpus(self, corpus, batch_size,**kwargs):
        corpus_embeddings = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        with torch.no_grad():
            for start_idx in trange(0, len(sentences), batch_size):
                sub_sentences = sentences[start_idx : start_idx + self.batch_size]
                ctx_input = self.data_collator(sub_sentences)
                corpus_embeddings.append(self.encoder(ctx_input["input_ids"].to(self.encoder.device), ctx_input["attention_mask"].to(self.encoder.device)))

        co_corpus_embeddings = torch.cat(corpus_embeddings)

        return co_corpus_embeddings.cpu()
    
    def encode_queries(self, queries, batch_size,**kwargs):
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                query_input = self.data_collator(queries[start_idx:start_idx + batch_size])
                query_embeddings.append(self.encoder(query_input["input_ids"].to(self.encoder.device), query_input["attention_mask"].to(self.encoder.device)))

        co_query_embeddings = torch.cat(query_embeddings)

        return co_query_embeddings.cpu()
    
    def eval(self):
        self.encoder.eval() 
        self.encoder.to("cuda:0") 

class LongtrieverRetriever(BertRetriever):
    def forward(self, query_input_ids, query_attention_mask, corpus_input_ids, corpus_attention_mask):
        query_embeddings = self.encoder(query_input_ids, query_attention_mask)
        corpus_embeddings = self.encoder(corpus_input_ids, corpus_attention_mask)
        co_query_embeddings = torch.cat(self._gather_tensor(query_embeddings.contiguous()))
        co_corpus_embeddings = torch.cat(self._gather_tensor(corpus_embeddings.contiguous()))
        co_loss = self.compute_contrastive_loss(co_query_embeddings, co_corpus_embeddings)
        return (co_loss,)