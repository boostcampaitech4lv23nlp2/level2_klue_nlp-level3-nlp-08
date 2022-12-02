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
        self.dropout = nn.Dropout(p=0.5)

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
        y = torch.sigmoid(y)
        y = self.dropout(y)
        logits = self.linear2(y) # (Batch, 300)

        return {'logits':logits}

class SpecificModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(self.model_name)

        self.model = AutoModel.from_pretrained(self.model_name)
        # cls layer
        self.cls_layers = nn.Sequential(
                            nn.Linear(self.config.hidden_size, self.config.hidden_size),
                            nn.GELU(),
                            nn.Dropout(0.1)
                            )

        #TODO sub obj 나누지말고 하나의 layers로 통합해보기
        self.sub_layers = nn.Sequential(
                            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
                            )
        self.obj_layers = nn.Sequential(
                            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
                            )

        self.output_layer = nn.Sequential(
                            nn.Linear(self.config.hidden_size * 3, self.config.hidden_size),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(self.config.hidden_size, 30)
                            )

    def forward(self, input_ids, token_type_ids, attention_mask, entity_loc_ids):
        sub_mask = (entity_loc_ids == 1).type(torch.long)
        sub_mask_unsqueeze = sub_mask.unsqueeze(1)
        obj_mask = (entity_loc_ids == 2).type(torch.long)
        obj_mask_unsqueeze = obj_mask.unsqueeze(1)
        sub_lengths = (sub_mask != 0).sum(dim=1).unsqueeze(1)
        obj_lengths = (obj_mask != 0).sum(dim=1).unsqueeze(1)
        
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # b, d
        pooler_output = outputs['pooler_output']
        pooler_output = self.cls_layers(pooler_output)
        # b l d
        outputs = outputs['last_hidden_state']

        # [b 1 len] [b len d] -> [b 1, d] -> [b, d]
        sub_sum_tensors = torch.bmm(sub_mask_unsqueeze.float(), outputs).squeeze(1)
        obj_sum_tensors = torch.bmm(obj_mask_unsqueeze.float(), outputs).squeeze(1)

        sub_averages = sub_sum_tensors.float() / sub_lengths.float()
        obj_averages = obj_sum_tensors.float() / obj_lengths.float()

        sub_output = self.sub_layers(sub_averages)
        obj_output = self.obj_layers(obj_averages)
        
        # b d*3
        outputs = torch.cat((pooler_output, sub_output, obj_output), dim=1)
        # b 30
        outputs = self.output_layer(outputs)

        return {'logits': outputs}
