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