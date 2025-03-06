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

import json

from beir.retrieval.models.pooling import cls_pooling, eos_pooling, mean_pooling
from beir.retrieval.models.util import extract_corpus_sentences

import os

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


class BiEncoder:
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
        **kwargs,
    ):
        self.sep = sep
        if peft_model_path:
            
            # Support shared and independent models
            if isinstance(peft_model_path, str):
                self.q_model, base_model_path = get_peft_model(peft_model_path, **kwargs)
                self.doc_model = self.q_model

            elif isinstance(peft_model_path, tuple):
                self.q_model, base_model_path = get_peft_model(peft_model_path[0], **kwargs)
                self.doc_model, base_model_path = get_peft_model(peft_model_path[1], **kwargs)
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        else:
            # Support shared and independent models
            if isinstance(model_path, str):
                self.q_model = AutoModel.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=kwargs.get("torch_dtype", "auto"),
                    trust_remote_code=True,
                    attn_implementation=kwargs.get("attn_implementation", "eager"),
                    cache_dir=kwargs.get("cache_dir", None),
                )
                self.doc_model = self.q_model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

            elif isinstance(model_path, tuple):
                self.q_model = AutoModel.from_pretrained(
                    model_path[0],
                    device_map="auto",
                    torch_dtype=kwargs.get("torch_dtype", "auto"),
                    trust_remote_code=True,
                    attn_implementation=kwargs.get("attn_implementation", "eager"),
                    cache_dir=kwargs.get("cache_dir", None),
                )
                self.doc_model = AutoModel.from_pretrained(
                    model_path[1],
                    device_map="auto",
                    torch_dtype=kwargs.get("torch_dtype", "auto"),
                    trust_remote_code=True,
                    attn_implementation=kwargs.get("attn_implementation", "eager"),
                    cache_dir=kwargs.get("cache_dir", None),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path[0], use_fast=True)
                
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.max_length = max_length if max_length else self.tokenizer.model_max_length
        self.normalize = normalize  # Normalize the embeddings
        self.append_eos_token = append_eos_token  # Add eos token to the input
        self.train()
        if pooling not in ["cls", "mean", "eos"]:
            raise ValueError("Supported Pooling techniques should be either 'cls', 'mean' or 'eos'")
        self.pooling_func = POOL_FUNC[pooling]

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

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        query_embeddings = []

        context_manager = torch.no_grad() if not self.q_model.training else torch.enable_grad()
        with context_manager:
            for start_idx in range(0, len(queries), batch_size):
                sub_queries = [self.query_prefix + query for query in queries[start_idx : start_idx + batch_size]]
                if self.append_eos_token:
                    query_input = self._append_eos_token(sub_queries)
                else:
                    query_input = self.tokenizer(sub_queries, truncation=True, padding=True, return_tensors="pt")

                # Move the input to the device
                query_input = query_input.to(self.q_model.device)
                query_output = self.q_model(**query_input)
                query_embeddings.append(self.pooling_func(query_output, query_input["attention_mask"]))

        query_embeddings = torch.cat(query_embeddings)

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        if self.q_model.training:
            return query_embeddings
        else:
            return query_embeddings.cpu()

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], batch_size: int = 8, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        corpus_embeddings = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        context_manager = torch.no_grad() if not self.doc_model.training else torch.enable_grad()
        with context_manager:            
            for start_idx in trange(0, len(sentences), batch_size):
                sub_sentences = [
                    self.doc_prefix + sentence for sentence in sentences[start_idx : start_idx + batch_size]
                ]
                if self.append_eos_token:
                    ctx_input = self._append_eos_token(sub_sentences)
                else:
                    ctx_input = self.tokenizer(sub_sentences, truncation=True, padding=True, return_tensors="pt")

                # Move the input to the device
                ctx_input = ctx_input.to(self.doc_model.device)
                ctx_output = self.doc_model(**ctx_input)
                corpus_embeddings.append(self.pooling_func(ctx_output, ctx_input["attention_mask"]))

            corpus_embeddings = torch.cat(corpus_embeddings)

            if self.normalize:
                corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)

            if self.doc_model.training:
                return corpus_embeddings
            else:
                return corpus_embeddings.cpu()

    def train(self):
        self.q_model.train()
        self.doc_model.train()
    
    def eval(self):
        self.q_model.eval()
        self.doc_model.eval()

    def save_pretrained(self, save_path: str, is_main_process = True, save_function = torch.save):
        os.makedirs(save_path, exist_ok=True)

        self.q_model.save_pretrained(os.path.join(save_path, "q_model"), is_main_process=is_main_process, save_function=save_function)

        if self.q_model is not self.doc_model:
            self.doc_model.save_pretrained(os.path.join(save_path, "doc_model"), is_main_process=is_main_process, save_function=save_function)

        self.tokenizer.save_pretrained(save_path)

        config = {
            "max_length": self.max_length,
            "sep": self.sep,
            "pooling": self.pooling_func.__name__.replace("_pooling", ""),
            "normalize": self.normalize,
            "append_eos_token": self.append_eos_token,
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
            "shared_encoder": self.q_model is self.doc_model,  # Store if encoders are shared
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load the bi-encoder model, tokenizer, and config."""
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Load models
        if config["shared_encoder"]:
            q_model = AutoModel.from_pretrained(os.path.join(load_path, "q_model"))
            doc_model = q_model  # Share the same model
        else:
            q_model = AutoModel.from_pretrained(os.path.join(load_path, "q_model"))
            doc_model = AutoModel.from_pretrained(os.path.join(load_path, "doc_model"))

        # Create instance
        instance = cls(
            model_path=None,  # We manually set models, so model_path is not needed
            max_length=config["max_length"],
            sep=config["sep"],
            pooling=config["pooling"],
            normalize=config["normalize"],
            append_eos_token=config["append_eos_token"],
            prompts={"query": config["query_prefix"], "passage": config["doc_prefix"]},
        )
        instance.q_model = q_model
        instance.doc_model = doc_model
        instance.tokenizer = tokenizer
        return instance