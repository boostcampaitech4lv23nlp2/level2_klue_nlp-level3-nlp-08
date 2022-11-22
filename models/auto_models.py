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

class RE_Model(nn.Module):
    def __init__(self,MODEL_NAME:str):
        super().__init__()
        self.MODEL_NAME = MODEL_NAME
        # plm 모델 설정
        self.plm = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME,num_labels = 30)
    
    def forward(self,**batch):
        outputs = self.plm(**batch)
        return outputs

class CNN_Model(nn.Module):
    def __init__(self,MODEL_NAME):
        super().__init__()
        self.MODEL_NAME = MODEL_NAME
        self.model_config = AutoConfig.from_pretrained(self.MODEL_NAME)
        self.plm = AutoModel.from_pretrained(self.MODEL_NAME)
        self.cnn_layers = nn.ModuleList([nn.Conv1d(in_channels=self.model_config.hidden_size,out_channels=100,kernel_size=i) for i in range(2,8)])
        self.pooling_layers = nn.ModuleList([nn.MaxPool1d(256-i+1) for i in range(2,8)])
        self.linear = nn.Linear(600,300)
        self.linear2 = nn.Linear(300,30)

    def forward(self,**batch):
        inputs = {'input_ids':batch.get('input_ids'),'token_type_ids':batch.get('token_type_ids'),'attention_mask':batch.get('attention_mask')}
        y = self.plm(**inputs)
        y = y.last_hidden_state
        y= y.transpose(1,2)
        tmp = []
        for i in range(len(self.cnn_layers)):
            t = torch.tanh(self.cnn_layers[i](y))
            t = self.pooling_layers[i](t)
            tmp.append(t)

        y = torch.cat(tmp,axis=1).squeeze()

        logits = self.linear(y)
        logits = self.linear2(logits)

        return logits

        

