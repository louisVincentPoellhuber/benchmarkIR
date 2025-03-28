import torch
from transformers import BertModel, AutoModel
from transformers.models.bert.modeling_bert import BertEmbeddings,BertLayer,BertOnlyMLMHead,BertPreTrainedModel, BertConfig
from torch import Tensor, nn
from enhancedDecoder import BertLayerForDecoder
from typing import Optional


class LongtrieverConfig(BertConfig):
    model_type = "longtriever"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BlockLevelContextawareEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoding_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.information_exchanging_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

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
        # if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
        #     print("NaNs or Infs detected in hidden_states before BertLayer!")
        for i, layer_module in enumerate(self.text_encoding_layer):
            if i>0: # Every subsequent layer
                # hidden_states = torch.clamp(hidden_states, min=-1e4, max=1e4)  # Prevent large values
                layer_outputs = layer_module(hidden_states, attention_mask)
            else: # The first layer
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:,:,:,0] = -10000.0
                # hidden_states = torch.clamp(hidden_states, min=-1e4, max=1e4)  # Prevent large values
                layer_outputs = layer_module(hidden_states, temp_attention_mask)
                reduce_hidden_states=reduce_hidden_states[None,:,:].repeat(B,1,1)

            hidden_states = layer_outputs[0]

            hidden_states = hidden_states.view(B, N, L_, D)
            cls_hidden_states = hidden_states[:, :, 1, :].clone() # These are the CLS tokens: the block representations. If I copied it above, I could append it to the input and attend to them directly!
            
            reduce_hidden_states = torch.clamp(reduce_hidden_states, min=-1e18, max=1e18)
            reduce_cls_hidden_states=torch.cat([reduce_hidden_states,cls_hidden_states],dim=1) #[B,N+1,D] So it's the doc token [DOC] with every block's [CLS] token
            station_hidden_states = self.information_exchanging_layer[i](reduce_cls_hidden_states, node_mask)[0]
            reduce_hidden_states = station_hidden_states[:,:1,:]
            hidden_states[:, :, 0, :] = station_hidden_states[:,1:,:]
            hidden_states = hidden_states.view(B * N, L_, D)

        return (reduce_hidden_states, hidden_states, )



class Longtriever(BertModel):   
    config_class = LongtrieverConfig
    base_model_prefix = "longtriever"
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BlockLevelContextawareEncoder(config)
        self.doc_embeddings = nn.Embedding(1,config.hidden_size).weight #[1,D]

        # Initialize weights and apply final processing
        self.post_init()

    def get_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        #station_mask==0? for attention_mask==0
        station_mask = torch.ones((attention_mask.shape[0],1),dtype=attention_mask.dtype,device=attention_mask.device) # [B*N,1]
        attention_mask = torch.cat([station_mask,attention_mask],dim=1) # [B*N,1+L]
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask,
        sentence_mask=None,
        return_last_hiddens=False, 
        token_type_ids = None
    ):
        if sentence_mask is None:
            sentence_mask = (torch.sum(attention_mask,dim=-1)>0).to(dtype=attention_mask.dtype)
        # sentence = blocks
        sentence_mask = torch.cat([torch.ones_like(sentence_mask[:,:1]),sentence_mask],dim=1)

        input_shape = input_ids.size()
        batch_size, sent_num, seq_length = input_shape
        # So they concatenate all the blocks together
        input_ids = input_ids.view(batch_size*sent_num,seq_length)
        attention_mask = attention_mask.view(batch_size*sent_num,seq_length)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask) #[B*N,1,1,1+L] L is sequence length. [24, 1, 1,513]
        extended_sentence_mask = (1.0 - sentence_mask[:, None, None, :]) * -10000.0

        embedding_output = self.embeddings(input_ids=input_ids)
        station_placeholder = torch.zeros((embedding_output.shape[0], 1, embedding_output.shape[-1]),dtype=embedding_output.dtype,device=embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # [B*N,1+L,D]

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            reduce_hidden_states=self.doc_embeddings,
            node_mask=extended_sentence_mask,
        )
        text_vec = encoder_outputs[0].squeeze(1)

        # It should be a BaseModelOutputWithPoolingAndCrossAttentions
        if not return_last_hiddens:
            return text_vec
        else:
            last_hiddens=encoder_outputs[1].view(batch_size,sent_num,seq_length+1,self.config.hidden_size)
            return (text_vec, last_hiddens) #[B,D] and #[B,N,L+1,D]
    
class LongtrieverForPretraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = Longtriever(config)
        self.cls = BertOnlyMLMHead(config)

        self.decoder_embeddings = self.bert.embeddings
        self.c_head = BertLayerForDecoder(config)
        self.c_head.apply(self._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.post_init()

    def forward(self, encoder_input_ids_batch, encoder_attention_mask_batch, encoder_labels_batch, decoder_input_ids_batch,
                decoder_matrix_attention_mask_batch, decoder_labels_batch,):

        batch_size, sent_num, seq_length = encoder_input_ids_batch.shape

        text_vec, last_hiddens=self.bert(encoder_input_ids_batch,encoder_attention_mask_batch,return_last_hiddens=True)
        last_hiddens=last_hiddens[:,:,1:,:]
        _, encoder_mlm_loss = self.mlm_loss(last_hiddens,encoder_labels_batch)

        cls_hiddens = last_hiddens[:, :, :1, :] # [B,N,1,D]
        doc_embeddings = text_vec[:,None,None,:].repeat(1,sent_num,1,1) #[B,N,1,D]

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids_batch.view(batch_size*sent_num,seq_length)).view(batch_size,sent_num,seq_length,self.config.hidden_size) # [B,N,L,D]
        hiddens = torch.cat([doc_embeddings,cls_hiddens, decoder_embedding_output[:, :, 1:, :]], dim=2).view(batch_size*sent_num,seq_length+1,self.config.hidden_size) #[B*N,L+1,D]

        decoder_position_ids = self.bert.embeddings.position_ids[:, :seq_length]
        decoder_position_embeddings = self.bert.embeddings.position_embeddings(decoder_position_ids) # [1,L,D]
        query = (decoder_position_embeddings[:,None,:,:] + doc_embeddings).view(batch_size*sent_num,seq_length,self.config.hidden_size) #[B*N,L,D]

        decoder_matrix_attention_mask_batch=decoder_matrix_attention_mask_batch.view(batch_size*sent_num,seq_length,seq_length) #[B*N,L,L]
        decoder_matrix_attention_mask_batch=torch.cat([decoder_matrix_attention_mask_batch.new_full((batch_size*sent_num,seq_length,1),1),decoder_matrix_attention_mask_batch],dim=2) #[B*N,L,L+1]
        matrix_attention_mask = self.get_extended_attention_mask(
            decoder_matrix_attention_mask_batch,
            decoder_matrix_attention_mask_batch.shape,
            decoder_matrix_attention_mask_batch.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        _, decoder_mlm_loss = self.mlm_loss(hiddens, decoder_labels_batch)

        return (encoder_mlm_loss+decoder_mlm_loss, )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss
    






class HierarchicalLongtrieverConfig(BertConfig):
    model_type = "hierarchical_longtriever"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HierarchicalLongtrieverEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)

    def add_blockwise_cls_tokens(self, inputs_embeds, attention_mask):
        cls_embeds = inputs_embeds[:, :, 0, :]
        cls_attention_mask = attention_mask[:, :, 0] if attention_mask != None else None
        block_input_embeds = []
        block_attention_mask = []
        for i in range(inputs_embeds.shape[1]):
            current_block_input_embeds = inputs_embeds[:, i, :, :]
            pre_cls_embeds = cls_embeds[:, 0:i, :]
            post_cls_embeds = cls_embeds[:, i:-1, :]
            hier_block_input_embeds = torch.cat([pre_cls_embeds, current_block_input_embeds, post_cls_embeds], dim=1)
            block_input_embeds.append(hier_block_input_embeds)

            if attention_mask!=None:
                current_block_attention_mask = attention_mask[:, i, :]
                pre_cls_attention_mask = cls_attention_mask[:, 0:i]
                post_cls_attention_mask = cls_attention_mask[:, i:-1]
                hier_block_attention_mask = torch.cat([pre_cls_attention_mask, current_block_attention_mask, post_cls_attention_mask], dim=1)
                block_attention_mask.append(hier_block_attention_mask)
        
        block_input_embeds = torch.stack(block_input_embeds, dim=1)
        block_attention_mask = torch.stack(block_attention_mask, dim=1)

        return block_input_embeds, block_attention_mask
    

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
        seq_length = input_shape[2] + input_shape[1] - 1

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], input_shape[1], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # TODO: maybe skip if N==1
        if input_shape[1]>1:
            block_input_embeds, block_attention_mask = self.add_blockwise_cls_tokens(inputs_embeds, attention_mask)
        else:
            block_input_embeds = inputs_embeds
            block_attention_mask = attention_mask

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
        pre_indices = torch.tril_indices(N, L_)
        mask = pre_indices[1] != 0
        filtered_indices = pre_indices[:, mask]

        extended_cls_tokens = cls_hidden_states.repeat(N, 1, 1, 1).view(B, N, N, D)
        pre_cls_indices = torch.tril_indices(N, N, offset=-1)

        hidden_states[:, filtered_indices[0], filtered_indices[1], :] = extended_cls_tokens[:, pre_cls_indices[0], pre_cls_indices[1], :]

        # post
        post_indices = torch.triu_indices(N, L_, offset=L_-N+1)
        mask = post_indices[1] != hidden_states.shape[1] - 1
        filtered_indices = post_indices[:, mask]

        post_cls_indices = torch.triu_indices(N, N, offset=1)

        hidden_states[:, filtered_indices[0], filtered_indices[1], :] = extended_cls_tokens[:, post_cls_indices[0], post_cls_indices[1], :]

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
            cls_hidden_states = hidden_states[:, torch.arange(N), torch.arange(N)+1, :].clone() # Diagonal now

            if N>1: # Pointless to do for small document and queries
                hidden_states = self.update_blockwise_cls_tokens(cls_hidden_states, hidden_states, B, N, L_, D)

            reduce_hidden_states = torch.clamp(reduce_hidden_states, min=-1e18, max=1e18)
            reduce_cls_hidden_states=torch.cat([reduce_hidden_states,cls_hidden_states],dim=1) #[B,N+1,D] So it's the doc token [DOC] with every block's [CLS] token
            station_hidden_states = self.information_exchanging_layer[i](reduce_cls_hidden_states, node_mask)[0]
            reduce_hidden_states = station_hidden_states[:,:1,:]
            hidden_states[:, :, 0, :] = station_hidden_states[:,1:,:] # Replace the placeholder DOC tokens by the actual DOC tokens
            hidden_states = hidden_states.view(B * N, L_, D)

        return (reduce_hidden_states, hidden_states, )


class HierarchicalLongtriever(Longtriever):
    config_class = HierarchicalLongtrieverConfig
    base_model_prefix = "hierarchical_longtriever"
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = HierarchicalLongtrieverEmbeddings(config)
        self.encoder = BlockLevelHierarchicalContextawareEncoder(config)

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
    
