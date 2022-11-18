import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
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