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

class r_roberta_Classifier(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=3, dr_rate=0.0):
        super(r_roberta_Classifier, self).__init__()
        self.roberta = roberta
        self.dr_rate = dr_rate

        self.cls_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.sentence_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.label_classifier = FCLayer(hidden_size//2 * 3, num_classes, self.dr_rate, False)

    def forward(self, token_ids, attention_mask, segment_ids=None):
        out = self.roberta(input_ids=token_ids, attention_mask=attention_mask)[0]
        
        sentence_end_position = torch.where(token_ids == 2)[1]
        sent1_end, sent2_end = sentence_end_position[0], sentence_end_position[1]
        
        cls_vector = out[:, 0, :] # take <s> token (equiv. to [CLS])
        prem_vector = out[:,1:sent1_end]              # Get Premise vector
        hypo_vector = out[:,sent1_end+1:sent2_end]    # Get Hypothesis vector

        prem_vector = torch.mean(prem_vector, dim=1) # Average
        hypo_vector = torch.mean(hypo_vector, dim=1)

        
        # Dropout -> tanh -> fc_layer (Share FC layer for premise and hypothesis)
        cls_embedding = self.cls_fc(cls_vector)
        prem_embedding = self.sentence_fc(prem_vector)
        hypo_embedding = self.sentence_fc(hypo_vector)
        
        # Concat -> fc_layer
        concat_embedding = torch.cat([cls_embedding, prem_embedding, hypo_embedding], dim=-1)
        
        return self.label_classifier(concat_embedding)

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)