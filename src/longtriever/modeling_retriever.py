from modeling_utils import *
import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.nn.functional as F
from beir.retrieval.models.util import extract_corpus_sentences
from tqdm import trange
from data_handler import DataCollatorForEvaluatingLongtriever

def compute_contrastive_loss( co_query_embeddings, co_corpus_embeddings):
    similarities_1 = torch.matmul(co_query_embeddings, co_corpus_embeddings.transpose(0, 1))
    similarities_2 = torch.matmul(co_query_embeddings, co_query_embeddings.transpose(0, 1))
    similarities_2.fill_diagonal_(float('-inf'))
    similarities=torch.cat([similarities_1,similarities_2],dim=1)
    labels=torch.arange(similarities.shape[0],dtype=torch.long,device=similarities.device)
    co_loss = F.cross_entropy(similarities, labels) * dist.get_world_size()
    return co_loss

def compute_cross_entropy_loss(co_query_embeddings, co_corpus_embeddings):
    similarities = torch.matmul(co_query_embeddings, co_corpus_embeddings.transpose(0, 1))

    labels=torch.arange(similarities.shape[0],dtype=torch.long,device=similarities.device)
    co_loss = F.cross_entropy(similarities, labels) * dist.get_world_size()
    return co_loss


LOSS_FUNCTIONS = {
    "contrastive": compute_contrastive_loss,
    "cross_entropy": compute_cross_entropy_loss,
}


class BertRetriever(nn.Module):
    def __init__(self,
                 model,
                 data_collator=DataCollatorForEvaluatingLongtriever("bert-base-uncased", 512, 512, 8), 
                 normalize=False, 
                 loss_function="contrastive"):
        super().__init__()
        self.encoder=model
        # self.batch_size=batch_size
        self.sep=" [SEP] "
        self.data_collator = data_collator
        self.normalize = normalize
        self.loss_fct = LOSS_FUNCTIONS[loss_function]

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
        co_loss = self.loss_fct(co_query_embeddings, co_corpus_embeddings)
        return (co_loss,)

    
    def encode_corpus(self, corpus, batch_size, **kwargs):
        corpus_embeddings = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        with torch.no_grad():
            for start_idx in trange(0, len(sentences), batch_size):

                sub_sentences = sentences[start_idx : start_idx + batch_size]
            
                corpus_embeddings.append(self.tokenize(sub_sentences))

        co_corpus_embeddings = torch.cat(corpus_embeddings)
        
        # NOTE: It's actually called normalize_embeddings, but I want self.normalize to take priority
        if kwargs.get("normalize", self.normalize):
            corpus_embeddings = F.normalize(co_corpus_embeddings, p=2, dim=1)

        return co_corpus_embeddings.cpu()
    
    
    def encode_queries(self, queries, batch_size,**kwargs):
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                sub_queries = queries[start_idx:start_idx + batch_size]
                query_embeddings.append(self.tokenize(sub_queries))

        co_query_embeddings = torch.cat(query_embeddings)

        if kwargs.get("normalize", self.normalize):
            query_embeddings = F.normalize(co_query_embeddings, p=2, dim=1)

        return co_query_embeddings.cpu()
    
    def tokenize(self, sub_input):
        ctx_input = self.data_collator(sub_input)
        ctx_input_ids = ctx_input["input_ids"].to(self.encoder.device)
        ctx_attention_mask = ctx_input["attention_mask"].to(self.encoder.device)
        ctx_outputs = self.encoder(ctx_input_ids, ctx_attention_mask).last_hidden_state[:, 0]

        return ctx_outputs
    
    def eval(self):
        self.encoder.eval() 
        self.encoder.to("cuda:0") 
        self.training = False

class LongtrieverRetriever(BertRetriever):
    def forward(self, query_input_ids, query_attention_mask, corpus_input_ids, corpus_attention_mask):
        query_embeddings = self.encoder(query_input_ids, query_attention_mask)
        corpus_embeddings = self.encoder(corpus_input_ids, corpus_attention_mask)
        co_query_embeddings = torch.cat(self._gather_tensor(query_embeddings.contiguous()))
        co_corpus_embeddings = torch.cat(self._gather_tensor(corpus_embeddings.contiguous()))
        co_loss = self.loss_fct(co_query_embeddings, co_corpus_embeddings)
        return (co_loss,)
    
    def tokenize(self, sub_input):
        ctx_input = self.data_collator(sub_input)
        ctx_input_ids = ctx_input["input_ids"].to(self.encoder.device)
        ctx_attention_mask = ctx_input["attention_mask"].to(self.encoder.device)
        ctx_outputs = self.encoder(ctx_input_ids, ctx_attention_mask) # No last_hidden_state

        return ctx_outputs