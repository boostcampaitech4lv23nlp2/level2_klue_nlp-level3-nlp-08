import torch
import torch.nn as nn
from transformers import AutoModel,AutoConfig


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim,use_activation = True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RoBERTa_BiLSTM(nn.Module):
    def __init__(self,MODEL_NAME):
        super(RoBERTa_BiLSTM, self).__init__()
        self.MODEL_NAME = MODEL_NAME
        self.Backbone = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model_config = AutoConfig.from_pretrained(self.MODEL_NAME)
        self.hidden_size = self.model_config.hidden_size
        self.num_labels = 30

        self.lstm= nn.LSTM(input_size= self.hidden_size, hidden_size= self.hidden_size, num_layers= 2, dropout= 0.2,
                           batch_first= True, bidirectional= True)
        self.label_classifier = FCLayer(
            self.hidden_size * 2,
            30,
            use_activation=False,
        )

    def forward(self,**batch):
        inputs = {'input_ids':batch.get('input_ids'),'token_type_ids':batch.get('token_type_ids'),'attention_mask':batch.get('attention_mask')}
        outputs = self.Backbone(**inputs)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        # LSTM
        lstm_outputs,(lstm_hidden_state,lstm_cell_state) = self.lstm(sequence_output)
        cat_hidden= torch.cat((lstm_hidden_state[0], lstm_hidden_state[1]), dim= 1) # B, h*2
        logits = self.label_classifier(cat_hidden)

        return {'logits':logits}