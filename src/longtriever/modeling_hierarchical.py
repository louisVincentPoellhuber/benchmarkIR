import torch
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings,BertLayer,BertOnlyMLMHead,BertPreTrainedModel,BertConfig
from torch import Tensor, nn
from typing import Optional
from modeling_longtriever import *


class HierarchicalLongtrieverConfig(BertConfig):
    model_type = "hierarchical_longtriever"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HierarchicalLongtrieverEmbeddings(BertEmbeddings):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        ablation_config = kwargs.get("ablation_config", {"separators": False, "segments": False})
        self.add_segments = ablation_config["segments"]
        self.add_separators = ablation_config["separators"]

    def add_blockwise_cls_tokens(self, inputs_embeds, attention_mask, token_type_ids):
        cls_embeds = inputs_embeds[:, :, 0, :]
        cls_attention_mask = attention_mask[:, :, 0] if attention_mask != None else None
        block_input_embeds = []
        block_attention_mask = []
        block_token_type_ids = []
        for i in range(inputs_embeds.shape[1]): # the number of blocks
            # Input Embeddings
            current_block_input_embeds = inputs_embeds[:, i, :, :] # Input embeddings for block i
            current_cls_embeds = current_block_input_embeds[:, 0:1, :] # CLS token for block i
            pre_cls_embeds = cls_embeds[:, 0:i, :] # Take the CLS tokens from the previous blocks
            current_final_sep_embeds = current_block_input_embeds[:, -1:, :] # Final SEP token for block i, after padding
            post_cls_embeds = cls_embeds[:, i:-1, :] # Take the CLS tokens from the next blocks
            hier_block_input_embeds = torch.cat([current_cls_embeds, pre_cls_embeds, current_block_input_embeds[:, 1:-1, :], post_cls_embeds, current_final_sep_embeds], dim=1)
            block_input_embeds.append(hier_block_input_embeds)

            # Attention Mask
            current_block_attention_mask = attention_mask[:, i, :]
            current_cls_attention_mask = current_block_attention_mask[:, 0:1]
            pre_cls_attention_mask = current_cls_attention_mask.clone().repeat(1, i)
            current_final_sep_attention_mask = current_cls_attention_mask.clone() # If the token is CLS, it'll be 1, if it's PAD, it'll be 0
            post_cls_attention_mask = cls_attention_mask[:, i:-1]
            hier_block_attention_mask = torch.cat([current_cls_attention_mask, pre_cls_attention_mask, current_block_attention_mask[:, 1:-1], post_cls_attention_mask, current_final_sep_attention_mask], dim=1)
            block_attention_mask.append(hier_block_attention_mask)

            # Token Type IDs
            current_token_type_ids = token_type_ids[:, i, :]
            current_cls_token_type_ids = current_token_type_ids[:, 0:1]
            current_final_sep_token_type_ids = current_token_type_ids[:, -1:]
            pre_token_type_ids = torch.tensor(1, dtype=current_token_type_ids.dtype, device=current_token_type_ids.device).repeat(current_token_type_ids.shape[0], i)
            post_token_type_ids = torch.tensor(1, dtype=current_token_type_ids.dtype, device=current_token_type_ids.device).repeat(current_token_type_ids.shape[0], token_type_ids.shape[1] - i - 1)
            hier_token_type_ids = torch.cat([current_cls_token_type_ids, pre_token_type_ids, current_token_type_ids[:, 1:-1], post_token_type_ids, current_final_sep_token_type_ids], dim=1)
            block_token_type_ids.append(hier_token_type_ids)

        block_input_embeds = torch.stack(block_input_embeds, dim=1)
        block_attention_mask = torch.stack(block_attention_mask, dim=1)
        block_token_type_ids = torch.stack(block_token_type_ids, dim=1)

        return block_input_embeds, block_attention_mask, block_token_type_ids
    

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None, 
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # Infer block length from shapes here
        input_seq_length = input_shape[2]
        real_seq_length = input_shape[2] + input_shape[1] - 1

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : real_seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :input_seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], input_shape[1], input_seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if input_shape[1]>1:
            block_input_embeds, block_attention_mask, token_type_ids = self.add_blockwise_cls_tokens(inputs_embeds, attention_mask, token_type_ids)
        else:
            block_input_embeds = inputs_embeds
            block_attention_mask = attention_mask

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = block_input_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        if attention_mask!=None:
            return embeddings, block_attention_mask
        else:
            return embeddings

class BlockLevelHierarchicalContextawareEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoding_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.information_exchanging_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def update_blockwise_cls_tokens(self, cls_hidden_states, hidden_states, B, N, L_, D):
        # pre
        pre_indices = torch.tril_indices(N, L_, offset=1)
        mask = (pre_indices[1] != 0) & (pre_indices[1] != 1) 
        pre_filtered_indices = pre_indices[:, mask]

        extended_cls_tokens = cls_hidden_states.repeat(N, 1, 1, 1).view(B, N, N, D)
        pre_cls_indices = torch.tril_indices(N, N, offset=-1)

        hidden_states[:, pre_filtered_indices[0], pre_filtered_indices[1], :] = extended_cls_tokens[:, pre_cls_indices[0], pre_cls_indices[1], :]

        # post
        post_indices = torch.triu_indices(N, L_, offset=L_-N+1) 
        # mask = post_indices[1] != hidden_states.shape[1] - 1
        # post_filtered_indices = post_indices[:, mask]

        post_cls_indices = torch.triu_indices(N, N, offset=1)

        hidden_states[:, post_indices[0], post_indices[1], :] = extended_cls_tokens[:, post_cls_indices[0], post_cls_indices[1], :]

        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask,
        reduce_hidden_states,
        node_mask,
    ): # B = batch size, N = nb of blocks, L_ = sequence length, D = hidden size
        _, L_, D = hidden_states.shape
        B, _, _, N_ = node_mask.shape
        N=N_-1

        for i, layer_module in enumerate(self.text_encoding_layer):
            if i>0: # Every subsequent layer
                layer_outputs = layer_module(hidden_states, attention_mask)
            else: # The first layer
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:,:,:,0] = -10000.0
                layer_outputs = layer_module(hidden_states, temp_attention_mask)
                reduce_hidden_states=reduce_hidden_states[None,:,:].repeat(B,1,1) # repeat it for all 3 examples in batch

            hidden_states = layer_outputs[0]

            hidden_states = hidden_states.view(B, N, L_, D)
            cls_hidden_states = hidden_states[:, :, 1, :].clone()

            if N>1: # Pointless to do for small document and queries
                hidden_states = self.update_blockwise_cls_tokens(cls_hidden_states, hidden_states, B, N, L_, D)

            # reduce_hidden_states = torch.clamp(reduce_hidden_states, min=-1e18, max=1e18)
            reduce_cls_hidden_states=torch.cat([reduce_hidden_states,cls_hidden_states],dim=1) #[B,N+1,D] So it's the doc token [DOC] with every block's [CLS] token
            station_hidden_states = self.information_exchanging_layer[i](reduce_cls_hidden_states, node_mask)[0]
            reduce_hidden_states = station_hidden_states[:,:1,:]
            hidden_states[:, :, 0, :] = station_hidden_states[:,1:,:] # Replace the placeholder DOC tokens by the actual DOC tokens
            hidden_states = hidden_states.view(B * N, L_, D)

        return (reduce_hidden_states, hidden_states, )


class HierarchicalLongtriever(Longtriever):
    config_class = HierarchicalLongtrieverConfig
    base_model_prefix = "hierarchical_longtriever"
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = HierarchicalLongtrieverEmbeddings(config, **kwargs)
        self.encoder = BlockLevelHierarchicalContextawareEncoder(config, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask,
        sentence_mask=None,
        return_last_hiddens=False,
        token_type_ids=None
    ):
        if sentence_mask is None:
            sentence_mask = (torch.sum(attention_mask,dim=-1)>0).to(dtype=attention_mask.dtype)
        # sentence = blocks
        sentence_mask = torch.cat([torch.ones_like(sentence_mask[:,:1]),sentence_mask],dim=1)

        input_shape = input_ids.size()
        batch_size, sent_num, seq_length = input_shape
        seq_length += sent_num - 1
        # So they concatenate all the blocks together
        # input_ids = input_ids.view(batch_size*sent_num,seq_length)

        embedding_output, block_attention_mask = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        embedding_output = embedding_output.view(batch_size*sent_num,seq_length, embedding_output.shape[3])
        station_placeholder = torch.zeros((embedding_output.shape[0], 1, embedding_output.shape[-1]),dtype=embedding_output.dtype,device=embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # [B*N,1+L,D]
        
        block_attention_mask = block_attention_mask.view(batch_size*sent_num,seq_length)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(block_attention_mask) #[B*N,1,1,1+L] L is sequence length. [24, 1, 1,513]
        extended_sentence_mask = (1.0 - sentence_mask[:, None, None, :]) * -10000.0

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            reduce_hidden_states=self.doc_embeddings,
            node_mask=extended_sentence_mask,
        )
        text_vec = encoder_outputs[0].squeeze(1)

        if not return_last_hiddens:
            return text_vec
        else:
            last_hiddens=encoder_outputs[1].view(batch_size,sent_num,seq_length+1,self.config.hidden_size)
            return (text_vec, last_hiddens) #[B,D] and #[B,N,L+1,D]
    
