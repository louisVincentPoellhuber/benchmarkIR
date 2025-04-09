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
        batch_size: int = 16,
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
                    #torch_dtype=kwargs.get("torch_dtype", "auto"),
                    trust_remote_code=True,
                    #attn_implementation=kwargs.get("attn_implementation", "eager"),
                    #cache_dir=kwargs.get("cache_dir", None),
                    **kwargs
                )
                self.doc_model = self.q_model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                self.shared_model=True

            elif isinstance(model_path, tuple):
                self.q_model = AutoModel.from_pretrained(
                    model_path[0],
                    #torch_dtype=kwargs.get("torch_dtype", "auto"),
                    trust_remote_code=True,
                    #attn_implementation=kwargs.get("attn_implementation", "eager"),
                    #cache_dir=kwargs.get("cache_dir", None),
                    **kwargs
                )
                self.doc_model = AutoModel.from_pretrained(
                    model_path[1],
                    #torch_dtype=kwargs.get("torch_dtype", "auto"),
                    trust_remote_code=True,
                    #attn_implementation=kwargs.get("attn_implementation", "eager"),
                    #cache_dir=kwargs.get("cache_dir", None),
                    **kwargs
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path[0], use_fast=True)
                self.shared_model=False

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
        
        context_manager = torch.no_grad() if (not self.q_model.training) | self.shared_model else torch.enable_grad()

        with context_manager:
            if not self.q_model.training:
                for start_idx in trange(0, len(queries), self.batch_size):
                    sub_queries = [self.query_prefix + query for query in queries[start_idx : start_idx + self.batch_size]]
                    query_embeddings.append(self._encode_queries(sub_queries))
                query_embeddings = torch.cat(query_embeddings)
            else:
                sub_queries = [self.query_prefix + query for query in queries]
                query_embeddings = self._encode_queries(sub_queries)


        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        return query_embeddings if self.q_model.training else query_embeddings.cpu()

    def _encode_queries(self, sub_queries: list[str], **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        if self.append_eos_token:
            query_input = self._append_eos_token(sub_queries)
        else:
            query_input = self.tokenizer(sub_queries, truncation=True, padding=True, return_tensors="pt")

        # Move the input to the device
        query_input = query_input.to(self.q_model.device)
        query_output = self.q_model(**query_input)
       
        return self.pooling_func(query_output, query_input["attention_mask"])

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        corpus_embeddings = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        context_manager = torch.no_grad() if not self.doc_model.training else torch.enable_grad()

        with context_manager:            
            if not self.doc_model.training: # If evaluating, the batching loop happens HERE
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

        return corpus_embeddings if self.doc_model.training else corpus_embeddings.cpu()

    
    def _encode_corpus(self, sub_sentences: list[str], **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        if self.append_eos_token:
            ctx_input = self._append_eos_token(sub_sentences)
        else:
            ctx_input = self.tokenizer(sub_sentences, truncation=True, padding=True, return_tensors="pt")

        # Move the input to the device
        ctx_input = ctx_input.to(self.doc_model.device)
        ctx_output = self.doc_model(**ctx_input)
       
        return self.pooling_func(ctx_output, ctx_input["attention_mask"])

    def train(self):
        self.q_model.train()
        self.doc_model.train()
    
    def eval(self):
        self.q_model.eval()
        self.q_model.to("cuda:0") # Map to GPU. If the model is training, accelerate will deal with it. 
        self.doc_model.eval()
        self.doc_model.to("cuda:0")

    def save_pretrained(self, save_path: str, is_main_process = True, save_function = torch.save, accelerator = None):
        if not is_main_process:
            return  
        os.makedirs(save_path, exist_ok=True)

        if accelerator is not None:
            self.q_model = accelerator.unwrap_model(self.q_model)
            self.doc_model = accelerator.unwrap_model(self.doc_model)

        q_model_path = os.path.join(save_path, "q_model")
        self.q_model.save_pretrained(q_model_path, is_main_process=is_main_process, save_function=save_function)
        self.tokenizer.save_pretrained(q_model_path) 

        doc_model_path = os.path.join(save_path, "doc_model")
        if self.q_model is not self.doc_model:
            self.doc_model.save_pretrained(doc_model_path, is_main_process=is_main_process, save_function=save_function)
            self.tokenizer.save_pretrained(doc_model_path)

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
        # tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Load models
        if config["shared_encoder"]:
            # q_model = AutoModel.from_pretrained(os.path.join(load_path, "q_model"))
            # doc_model = q_model  # Share the same model
            model_path = os.path.join(load_path, "q_model")
        else:
            # q_model = AutoModel.from_pretrained(os.path.join(load_path, "q_model"))
            # doc_model = AutoModel.from_pretrained(os.path.join(load_path, "doc_model"))
            model_path = (os.path.join(load_path, "q_model"), os.path.join(load_path, "doc_model")) # Already a tuple!

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
        # instance.q_model = q_model
        # instance.doc_model = doc_model
        # instance.tokenizer = tokenizer
        return instance
    
    def save_checkpoint(self, save_path: str, accelerator):
        if accelerator.is_main_process:
            # Create checkpoints directory
            ckpt_path = os.path.join(save_path, "checkpoints")
            os.makedirs(ckpt_path, exist_ok=True)
            
            # Generate unique checkpoint ID
            ckpt_id = len(os.listdir(ckpt_path))
            checkpoint_dir = os.path.join(ckpt_path, f"checkpoint-{ckpt_id}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            accelerator.save_state(checkpoint_dir)

    def load_checkpoint(self, load_path: str, optim=None, scheduler=None, 
                        accelerator=None, load_optimizer=True, load_scheduler=True):
         
        accelerator.load_state(load_path)

        return accelerator
    

    def accelerate_model(self, optim, dataloader, scheduler, accelerator):
        # q_model = self.q_model
        # doc_model = self.doc_model
        # q_model, doc_model, optim, dataloader, scheduler = accelerator.prepare(
        #     q_model, doc_model, optim, dataloader, scheduler
        # )
        # self.q_model = q_model
        # self.doc_model = doc_model

        optim, dataloader, scheduler = accelerator.prepare(optim, dataloader, scheduler)
        self.q_model = accelerator.prepare(self.q_model)
        self.doc_model = accelerator.prepare(self.doc_model)

        return optim, dataloader, scheduler


    # def encode_tokenized_queries(self, query_input,**kwargs) -> list[Tensor] | np.ndarray | Tensor:
    #     query_embeddings = []

    #     context_manager = torch.no_grad() if not self.q_model.training else torch.enable_grad()
    #     with context_manager:
    #         query_output = self.q_model(**query_input)
    #         query_embeddings = self.pooling_func(query_output, query_input["attention_mask"])

    #         if self.normalize:
    #             query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    #         if self.q_model.training:
    #             return query_embeddings
    #         else:
    #             return query_embeddings.cpu()

    # def encode_tokenized_corpus(
    #     self, ctx_input, pooling_layer = True, **kwargs
    # ) -> list[Tensor] | np.ndarray | Tensor:
    #     corpus_embeddings = []

    #     context_manager = torch.no_grad() if not self.doc_model.training else torch.enable_grad()
    #     with context_manager:            
    #         ctx_output = self.doc_model(**ctx_input)
    #         corpus_embeddings = self.pooling_func(ctx_output, ctx_input["attention_mask"]) if pooling_layer else ctx_output

    #         if self.normalize:
    #             corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)

    #         if self.doc_model.training:
    #             return corpus_embeddings
    #         else:
    #             return corpus_embeddings.cpu()

class LongBiEncoder(BiEncoder):
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
        max_block_length: int = 512,
        max_num_blocks: int = 8,
        align_right: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            max_length=max_length,
            sep=sep,
            pooling=pooling,
            normalize=normalize,
            prompts=prompts,
            append_eos_token=append_eos_token,
            peft_model_path=peft_model_path,
            batch_size=batch_size,
            **kwargs,
        )
        if self.q_model.base_model_prefix!=self.doc_model.base_model_prefix:
            self.hybrid = True
        else:
            self.hybrid=False
        self.align_right = align_right
        self.max_num_blocks=max_num_blocks

        self.model_type = kwargs.get("model_type", "longtriever")
        if self.model_type == "longtriever": # If normal longtriever model, we slice blocks of size 512 
            self.max_block_length=max_block_length
        elif self.model_type == "hierarchical": # If hierarchical longtriever, we have to slice blocks of size 512-7, to account for all other CLS tokens
            self.max_block_length=max_block_length-self.max_num_blocks+1

    def _encode_queries(self, sub_queries: list[str], **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        if self.hybrid:
            query_input = self.tokenizer(sub_queries, truncation=True, padding=True, return_tensors="pt")
            query_input = query_input.to(self.q_model.device)
            query_output = self.q_model(**query_input)

            return self.pooling_func(query_output, query_input["attention_mask"])
        else:
            query_input = self.batch_tokenize(sub_queries, self.q_model.device)
            query_output = self.q_model(**query_input)

            return query_output
    
    def _encode_corpus(self, sub_sentences: list[str], **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        ctx_input = self.batch_tokenize(sub_sentences, self.doc_model.device)
        ctx_output = self.doc_model(**ctx_input)
       
        return ctx_output
    
    def batch_tokenize(self, examples, device):
        input_ids_batch = []
        attention_mask_batch = []
        for e in examples:
            results=self.block_tokenize(e)
            input_ids_batch.append(results['input_ids_blocks'])
            attention_mask_batch.append(results['attention_mask_blocks'])


        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id, align_right=self.align_right).to(device)  # [B,N,L]
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0, align_right=self.align_right).to(device)  # [B,N,L]

        return {
            "input_ids": input_ids_batch, 
            "attention_mask": attention_mask_batch
        }

    def block_tokenize(self, string):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)
        
        block_len = self.max_block_length - self.tokenizer.num_special_tokens_to_add(False)
        input_ids_blocks = []
        attention_mask_blocks = []
        curr_block = []

        for input_ids_sent in results['input_ids']:
            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block: # This is for overflow
                input_ids_blocks.append(
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
                attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
                curr_block = []
                if len(input_ids_blocks) >= self.max_num_blocks:
                    break
            curr_block.extend(input_ids_sent)
        if len(curr_block) > 0:
            input_ids_blocks.append(
                torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
            
            attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
        
        input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id, align_right=self.align_right)
        attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0, align_right=self.align_right)
        
        return {
            "input_ids_blocks": input_ids_blocks,
            "attention_mask_blocks": attention_mask_blocks,
        }

    def save_pretrained(self, save_path: str, is_main_process = True, save_function = torch.save, accelerator = None):
        if not is_main_process:
            return  
        os.makedirs(save_path, exist_ok=True)

        if accelerator is not None:
            self.q_model = accelerator.unwrap_model(self.q_model)
            self.doc_model = accelerator.unwrap_model(self.doc_model)

        q_model_path = os.path.join(save_path, "q_model")
        self.q_model.save_pretrained(q_model_path, is_main_process=is_main_process, save_function=save_function)
        self.tokenizer.save_pretrained(q_model_path) 

        doc_model_path = os.path.join(save_path, "doc_model")
        if self.q_model is not self.doc_model:
            self.doc_model.save_pretrained(doc_model_path, is_main_process=is_main_process, save_function=save_function)
            self.tokenizer.save_pretrained(doc_model_path)

        config = {
            "max_length": self.max_length,
            "sep": self.sep,
            "pooling": self.pooling_func.__name__.replace("_pooling", ""),
            "normalize": self.normalize,
            "append_eos_token": self.append_eos_token,
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
            "shared_encoder": self.q_model is self.doc_model, 
            "max_block_length": self.max_block_length, 
            "max_num_blocks": self.max_num_blocks,
            "model_type": self.model_type
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)


    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load the bi-encoder model, tokenizer, and config."""
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)
        # Load models
        if config["shared_encoder"]:
            model_path = os.path.join(load_path, "q_model")
        else:
            model_path = (os.path.join(load_path, "q_model"), os.path.join(load_path, "doc_model")) # Already a tuple!

        # Create instance
        instance = cls(
            model_path=model_path,  # We manually set models, so model_path is not needed
            max_length=config["max_length"],
            sep=config["sep"],
            pooling=config["pooling"],
            normalize=config["normalize"],
            append_eos_token=config["append_eos_token"],
            prompts={"query": config["query_prefix"], "passage": config["doc_prefix"]},
            max_block_length=config["max_block_length"], 
            max_num_blocks=config["max_num_blocks"],
            model_type = config["model_type"]
        )
        return instance