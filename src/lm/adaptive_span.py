# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple


# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3


class AdaptiveMask(nn.Module):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.

    Args:
        max_size: maximum size (i.e. input dimension)
        ramp_size: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """

    def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
        nn.Module.__init__(self)
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(shape) + init_val, requires_grad=True)
        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
       #scaled_val = self.current_val * self._max_size
        mask = self.mask_template + self.current_val * self._max_size
        mask = mask / self._ramp_size + 1
        mask = mask.clamp(0, 1)

        if x.size(-1) < self._max_size:
            # the input could have been trimmed beforehand to save computation
            mask = mask[:, :, -x.size(-1):]
        x = x * mask
        return x

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.mean().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def clamp_param(self):
        """this need to be called after each update"""
        self.current_val.data.clamp_(0, 1)


class AdaptiveSpan(nn.Module):
    """Adaptive attention span for Transformerself.
    This module learns an attention span length from data for each
    self-attention head.

    Args:
        attn_span: maximum attention span
        adapt_span_loss: loss coefficient for the span length
        adapt_span_ramp: length of the masking ramp
        adapt_span_init: initial size ratio
        adapt_span_cache: adapt cache size to reduce memory usage
    """
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self._adapt_cache = config.adapt_span_cache
        self._max_span = config.attn_span
        self._loss_coeff = config.adapt_span_loss
        self._num_attention_heads = config.num_attention_heads
        self._mask = AdaptiveMask(max_size=self._max_span,
                                 ramp_size=config.adapt_span_ramp,
                                 init_val=config.adapt_span_init,
                                 shape=(config.num_attention_heads, 1, 1))

    def forward(self, attn, normalize=False):
        """mask attention with the right span"""
        # batch and head dimensions are merged together, so separate them first
        B = attn.size(0) # batch size = 16
        M = attn.size(1) # block size = 512
        #attn = attn.reshape(B // self._num_attention_heads, self._num_attention_heads, M, -1)

        attn = self._mask(attn)

        if normalize:
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # normalize so sum is 1
        
        #attn = attn.view(B, M, -1)
        #print(f"Final attention shape: {attn.shape}")
        return attn

    def get_trim_len(self):
        """how much of memory can be trimmed to reduce computation"""
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        # too fine granularity might be bad for the memory management
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def trim_memory(self, query, key, value, key_pe):
        """trim out unnecessary memory beforehand to reduce computation"""
        trim_len = self.get_trim_len()
        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self._max_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            # cache is too short! this happens when validation resumes
            # after a lot of updates.
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return key, value, key_pe

    def get_cache_size(self):
        """determine how long the cache should be"""
        if self._adapt_cache:
            trim_len = self.get_trim_len()
            # give a buffer of 64 steps since a span might increase
            # in future updates
            return min(self._max_span, self._max_span - trim_len + 64)
        else:
            return self._max_span

    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self._loss_coeff * self._max_span * self._mask.current_val.mean()

    def get_current_max_span(self):
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        self._mask.clamp_param()
 

def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size # size of a single head
        self.attn_span = config.attn_span
        self.mask = AdaptiveMask(max_size=config.attn_span,
                                 ramp_size=config.adapt_span_ramp,
                                 init_val=config.adapt_span_init,
                                 shape=(config.num_attention_heads, 1, 1))


        self.adaptive_span = AdaptiveSpan(config)
    
    # Size notations:
    # B = batch_size (16), H = hidden_size (768), M = block_size (512), L = attn_span (?)
    def forward(self, query, key, value): # 16x12x512x64<
            # [optional] trim out memory to reduce unnecessary computation
            #key, value, key_pe = self.adaptive_span.trim_memory(
             #   query, key, value, key_pe)

            # compute attention from context
            # B x M (dest) x (M+L) (src) --> B x M x M<
            attn_cont = torch.matmul(query, key.transpose(-1, -2))
            ##print(f"Context attention (QK^T): {attn_cont.shape}")
            #attn_cont = _unskew(attn_cont)  # B x M x L

            # compute the effect of position embedding
            # Removed, because we are using absolute positional encodings. <

            attn = attn_cont / math.sqrt(self.hidden_size)  # B x M X L_pos

            attn = F.softmax(attn, dim=-1)


            # trim attention lengths according to the learned span
            attn_cont = self.mask(attn_cont)

            out = torch.matmul(attn_cont, value)  # B x M x H

            return out

    def get_cache_size(self):
        return self.adaptive_span.get_cache_size()


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        assert config.hidden_size % config.num_attention_heads == 0
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attn = SeqAttention(config)
        self.proj_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_out = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_val = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def head_reshape(self, x):
        K = self.num_attention_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.num_attention_heads
        D = self.head_dim
        M = query.size(1)


        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        key_pe = self.head_reshape(key_pe)
        ##print("Key PE:")
        ##print(key_pe.shape)

        ##print("Query:")
        ##print(query.shape)
        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out



class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.inner_hidden_size)
        self.fc2 = nn.Linear(config.inner_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = MultiHeadSeqAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.ff = FeedForwardLayer(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
 
    # B = batch_size, H = hidden_size, M = block_size, L = attn_span
    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        if self.ff is not None:
            ff_out = self.ff(h)
            out = self.norm2(h + ff_out)  # B x M x H
        else:
            out = h
        return out


class TransformerSeq(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        # token embeddings
        self.config = config
        self.in_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.out_emb = nn.Linear(config.hidden_size, config.vocab_size)
        if config.dropout > 0:
            self.emb_dropout = nn.Dropout(config.dropout)
        else:
            self.emb_dropout = None
        # position embeddings
        self.key_pe = nn.Parameter(
            torch.randn(1, config.hidden_size // config.num_attention_heads, config.attn_span))

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(config)
            for _ in range(config.num_hidden_layers))

    def forward(self, x, h_cache, target=None):
        # x size = B x M
        block_size = x.size(1)
        h = self.in_emb(x)  # B x M x H
        if self.emb_dropout is not None:
            h = self.emb_dropout(h)

        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()
            if cache_size > block_size:
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + block_size:, :], h],
                    dim=1).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)  # B x M x H

        if self.emb_dropout is not None:
            h = self.emb_dropout(h)
       
        out = F.log_softmax(self.out_emb(h), dim=-1)
        dummy_loss = None

        return out, h_cache_next, dummy_loss
