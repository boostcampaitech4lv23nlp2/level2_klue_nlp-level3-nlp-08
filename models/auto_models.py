import torch
from torch import nn
from transformers import (AutoTokenizer, 
                          AutoConfig, 
                          AutoModelForSequenceClassification, 
                          Trainer, 
                          TrainingArguments, 
                          RobertaConfig, 
                          RobertaTokenizer, 
                          RobertaForSequenceClassification, 
                          BertTokenizer,
                          AutoModel)
from load_data import *
from typing import Optional, List, Tuple

class RE_Model(nn.Module):
    def __init__(self, MODEL_NAME:str):
        super().__init__()
        self.MODEL_NAME = MODEL_NAME
        # plm 모델 설정
        self.plm = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=30)
    
    def forward(self,**batch):
        outputs = self.plm(**batch)
        return outputs


class DAE_CT_Model(nn.Module):
    def __init__(self, MODEL_NAME:str, plm):
        super().__init__()
        self.num_labels = 30
        self.MODEL_NAME = MODEL_NAME
        # plm 모델 설정
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.activation = torch.nn.ELU(alpha=1.0)
        self.hidden_size = self.plm.plm.config.hidden_size
        self.final_linear = nn.Linear(256, self.num_labels)
        self.tying_linear = nn.Linear(self.hidden_size, 256, bias=False)
        self.bottleneck_model = nn.Sequential(
            torch.nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            torch.nn.ELU(alpha=1.0),
            nn.Linear(128, 256),
            torch.nn.ELU(alpha=1.0),
        )     
    def forward(self, **batch):
        ori_embedding = self.plm(**batch)['embedding']
        ori_logits = self.plm(**batch)['logits']
        bottleneck_input = self.tying_linear(ori_embedding)
        output = self.bottleneck_model(bottleneck_input)
        out_embedding = torch.matmul(output, self.tying_linear.weight)  
        logits = self.final_linear(bottleneck_input)
        return {'logits': logits, 'embedding': out_embedding, 'ori_embedding': ori_embedding}  

class CT_Model(nn.Module):
    def __init__(self, MODEL_NAME:str):
        super().__init__()
        self.num_labels = 30
        self.MODEL_NAME = MODEL_NAME
        # plm 모델 설정
        self.plm = AutoModel.from_pretrained(self.MODEL_NAME)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(self.plm.config.hidden_size, self.num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)
 
    def forward(self, **batch):
        pooler_output = self.plm(**batch).pooler_output
        dropout_output = self.dropout(pooler_output)
        linear_outputs = self.linear(dropout_output)

        return {'logits': linear_outputs, 'embedding': pooler_output}  
    
    
class CNN_Model(nn.Module):
    def __init__(self,MODEL_NAME):
        super().__init__()
        self.MODEL_NAME = MODEL_NAME
        self.model_config = AutoConfig.from_pretrained(self.MODEL_NAME) # hidden_size 
        self.plm = AutoModel.from_pretrained(self.MODEL_NAME,force_download = True)
        self.cnn_layers = nn.ModuleList([nn.Conv1d(in_channels=self.model_config.hidden_size,out_channels=100,kernel_size=i) for i in range(2,12)]) # 2~7 리셉티브 필드. -> 조밀 부터 멀리까지
        #self.cnn_layers2 = nn.ModuleList([nn.Conv1d(in_channels=300,out_channels=100,kernel_size=i) for i in [3,5,7]]) # 2~7 리셉티브 필드. -> 조밀 부터 멀리까지
        self.pooling_layers = nn.ModuleList([nn.MaxPool1d(256-i+1) for i in range(2,12)])
        self.linear1 = nn.Linear(1000,500)
        self.linear2 = nn.Linear(500,30)

    def forward(self,**batch):
        inputs = {'input_ids':batch.get('input_ids'),'token_type_ids':batch.get('token_type_ids'),'attention_mask':batch.get('attention_mask')}
        y = self.plm(**inputs)
        y = y.last_hidden_state
        y= y.transpose(1,2)  # y  ==  bert 거쳐서 나온  결과물.
        tmp = []
        for i in range(len(self.cnn_layers)):
            t = torch.relu(self.cnn_layers[i](y))
            #t = torch.tanh(self.cnn_layers2[i](t))
            t = self.pooling_layers[i](t)
            tmp.append(t)

        y = torch.cat(tmp,axis=1).squeeze() # (Batch , 600)

        y = self.linear1(y)
        y = torch.relu(y)
        logits = self.linear2(y) # (Batch, 300)

        return {'logits':logits}

        



class EntityModel(nn.Module):
    def __init__(self,MODEL_NAME):
        super().__init__()
        self.MODEL_NAME = MODEL_NAME
        # plm 모델 설정
        self.plm = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=30)
        self.config = AutoConfig.from_pretrained(self.MODEL_NAME)

        # entity
        self.pretrained_embeddings = self.plm.bert.embeddings
        self.embeddings = EntityEmbeddings(self.config)
        # set weight
        self._set_weights()

        # encoders
        self.encoder = self.plm.bert.encoder

        # pooler
        self.pooler = self.plm.bert.pooler

        # classifier
        self.layer = nn.Linear(self.config.hidden_size, 30)
        

    def _set_weights(self):
        self.embeddings.word_embeddings.weight = self.plm.bert.embeddings.word_embeddings.weight
        self.embeddings.token_type_embeddings.weight = self.plm.bert.embeddings.token_type_embeddings.weight
        self.embeddings.position_embeddings.weight = self.plm.bert.embeddings.position_embeddings.weight

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None, inputs_embeds=None, entity_ids=None, return_dict=False, use_cache=None):

        output_attentions = self.config.output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = 0

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        
        
        # embeddings
        embeddings_output = self.embeddings(input_ids = input_ids,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        inputs_embeds=inputs_embeds,
                        entity_ids=entity_ids)

        
        # encoder
        encoder_outputs = self.encoder(
            embeddings_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = self.layer(pooled_output)
        return {'outputs': outputs}


    def get_head_mask(
        self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ):
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):

        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def get_extended_attention_mask(
        self, attention_mask, input_shape, device = None, dtype = None
    ):
        
        if dtype is None:
            dtype = torch.float16

            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            
            
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


class EntityEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.entity_embeddings = nn.Embedding(2, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, entity_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        entity_embeddings = self.entity_embeddings(entity_ids)

        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings + entity_embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

