from __future__ import annotations

import importlib.util
import logging

if importlib.util.find_spec("peft") is not None:
    from peft import PeftConfig, PeftModel

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
import nltk
import json
from preprocessing.preprocess_utils import tensorize_batch
from beir.retrieval.models.pooling import cls_pooling, eos_pooling, mean_pooling
from beir.retrieval.models.util import extract_corpus_sentences
import os
from modeling_utils import log_message

logger = logging.getLogger(__name__)

POOL_FUNC = {"cls": cls_pooling, "mean": mean_pooling, "eos": eos_pooling}


def get_peft_model(peft_model_name: str, **kwargs) -> tuple[PeftModel, str]:
    config = PeftConfig.from_pretrained(peft_model_name)
    logger.info(f"Loading Auto Model from {config.base_model_name_or_path} for PEFT model")
    base_model = AutoModel.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        attn_implementation=kwargs.get("attn_implementation", "eager"),
        torch_dtype=kwargs.get("torch_dtype", "auto"),
        trust_remote_code=True,
        cache_dir=kwargs.get("cache_dir", None),
    )
    logger.info(f"Loading PEFT model from {peft_model_name}")
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload() # NOTE: Apparently this could break the gradient chain. Be careful if you decide to use PEFT. 
    return model, config.base_model_name_or_path


class SharedBiEncoder:
    def __init__(
        self,
        model_path: str | tuple = None,
        max_length: int = None,
        sep: str = " ",
        pooling: str = "mean",
        normalize: bool = False,
        prompts: dict[str, str] = None,
        append_eos_token: bool = False,
        peft_model_path: str = None,
        batch_size: int = 16,
        **kwargs,
    ):
        self.sep = sep
       
        self.encoder = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.shared_model=True

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        if max_length is not None:
            self.max_length = max_length 
            self.tokenizer.model_max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length
        self.normalize = normalize  # Normalize the embeddings
        self.append_eos_token = append_eos_token  # Add eos token to the input
        self.train()
        if pooling not in ["cls", "mean", "eos"]:
            raise ValueError("Supported Pooling techniques should be either 'cls', 'mean' or 'eos'")
        self.pooling_func = POOL_FUNC[pooling]
        self.batch_size = batch_size

        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")
        else:
            self.query_prefix = ""
            self.doc_prefix = ""

    def _append_eos_token(self, texts, pad_to_multiple_of: int = 16):
        """Tokenizes the input texts and pads the tokenized input to the max_length with the eos token"""
        collated_texts = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length - 1 if self.append_eos_token else self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        collated_texts["input_ids"] = [x + [self.tokenizer.eos_token_id] for x in collated_texts["input_ids"]]
        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return collated_texts

    def encode_queries(self, queries: list[str],**kwargs) -> list[Tensor] | np.ndarray | Tensor:
        query_embeddings = []
        
        context_manager = torch.no_grad() if (not self.encoder.training) else torch.enable_grad()

        with context_manager:
            if not self.encoder.training:
                for start_idx in trange(0, len(queries), self.batch_size):
                    sub_queries = [self.query_prefix + query for query in queries[start_idx : start_idx + self.batch_size]]
                    query_embeddings.append(self._encode_queries(sub_queries))
                query_embeddings = torch.cat(query_embeddings)
            else:
                sub_queries = [self.query_prefix + query for query in queries]
                query_embeddings = self._encode_queries(sub_queries)


        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        return query_embeddings if self.encoder.training else query_embeddings.cpu()

    def _encode_queries(self, sub_queries: list[str], **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        if self.append_eos_token:
            query_input = self._append_eos_token(sub_queries)
        else:
            query_input = self.tokenizer(sub_queries, truncation=True, padding=True, return_tensors="pt")

        # Move the input to the device
        query_input = query_input.to(self.encoder.device)
        query_output = self.encoder(**query_input)
       
        return self.pooling_func(query_output, query_input["attention_mask"])

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        corpus_embeddings = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        context_manager = torch.no_grad() if not self.encoder.training else torch.enable_grad()
    
        with context_manager:            
            if not self.encoder.training: # If evaluating, the batching loop happens HERE
                for start_idx in trange(0, len(sentences), self.batch_size):
                    sub_sentences = [
                        self.doc_prefix + sentence for sentence in sentences[start_idx : start_idx + self.batch_size]
                    ]
                    corpus_embeddings.append(self._encode_corpus(sub_sentences))
                corpus_embeddings = torch.cat(corpus_embeddings)
            else: # But if training, the batching loop happens in the main training loop instead
                sub_sentences = [self.doc_prefix + sentence for sentence in sentences]
                corpus_embeddings = self._encode_corpus(sub_sentences)


        if self.normalize:
            corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)

        return corpus_embeddings if self.encoder.training else corpus_embeddings.cpu()

    
    def _encode_corpus(self, sub_sentences: list[str], **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        if self.append_eos_token:
            ctx_input = self._append_eos_token(sub_sentences)
        else:
            ctx_input = self.tokenizer(sub_sentences, truncation=True, padding=True, return_tensors="pt")

        # Move the input to the device
        ctx_input = ctx_input.to(self.encoder.device)
        ctx_output = self.encoder(**ctx_input)
       
        return self.pooling_func(ctx_output, ctx_input["attention_mask"])

    def train(self):
        self.encoder.train()
        self.encoder.train()
    
    def eval(self):
        self.encoder.eval() 
        self.encoder.to("cuda:0") # Map to GPU. If the model is training, accelerate will deal with it. 

    def save_pretrained(self, save_path: str, is_main_process = True, save_function = torch.save, accelerator = None):
        if not is_main_process:
            return  
        os.makedirs(save_path, exist_ok=True)

        if accelerator is not None:
            self.encoder = accelerator.unwrap_model(self.encoder)

        encoder_path = os.path.join(save_path, "model")
        self.encoder.save_pretrained(encoder_path, is_main_process=is_main_process, save_function=save_function)
        self.tokenizer.save_pretrained(encoder_path) 


        config = {
            "max_length": self.max_length,
            "sep": self.sep,
            "pooling": self.pooling_func.__name__.replace("_pooling", ""),
            "normalize": self.normalize,
            "append_eos_token": self.append_eos_token,
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
            "shared_encoder": self.encoder is self.encoder,  # Store if encoders are shared
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)


    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load the bi-encoder model, tokenizer, and config."""
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)

        model_path = os.path.join(load_path, "model")

        # Create instance
        instance = cls(
            model_path=model_path,  # We manually set models, so model_path is not needed
            max_length=config["max_length"],
            sep=config["sep"],
            pooling=config["pooling"],
            normalize=config["normalize"],
            append_eos_token=config["append_eos_token"],
            prompts={"query": config["query_prefix"], "passage": config["doc_prefix"]},
        )
        return instance
    
    def save_checkpoint(self, save_path: str, accelerator, checkpoint_dir_name = "checkpoints"):
        if accelerator.is_main_process:
            # Create checkpoints directory
            ckpt_path = os.path.join(save_path, checkpoint_dir_name)
            os.makedirs(ckpt_path, exist_ok=True)
            
            # Generate unique checkpoint ID
            ckpt_id = len(os.listdir(ckpt_path))
            checkpoint_dir = os.path.join(ckpt_path, f"checkpoint-{ckpt_id}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            accelerator.save_state(checkpoint_dir)

    def load_checkpoint(self, load_path: str, accelerator=None):
         
        accelerator.load_state(load_path)

        return accelerator
    

    def accelerate_model(self, optim, dataloader, scheduler, accelerator):

        optim, dataloader, scheduler = accelerator.prepare(optim, dataloader, scheduler)
        self.encoder = accelerator.prepare(self.encoder)

        return optim, dataloader, scheduler

