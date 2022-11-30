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


class RBERT(nn.Module):
    def __init__(self,MODEL_NAME):
        super(RBERT, self).__init__()
        self.MODEL_NAME = MODEL_NAME
        self.Backbone = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model_config = AutoConfig.from_pretrained(self.MODEL_NAME)
        self.hidden_size = self.model_config.hidden_size
        self.num_labels = 30

        self.lstm= nn.LSTM(input_size= self.hidden_size, hidden_size= self.hidden_size, num_layers= 2, dropout= 0.2,
                           batch_first= True, bidirectional= True)
        self.cls_fc_layer = FCLayer(self.hidden_size, self.hidden_size) # 768 , 768 
        self.entity_fc_layer = FCLayer(self.hidden_size*2, self.hidden_size) # 768 , 768
        self.label_classifier = FCLayer(
            self.hidden_size * 3,
            30,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, e1_mask, e2_mask,**batch):
        inputs = {'input_ids':batch.get('input_ids'),'token_type_ids':batch.get('token_type_ids'),'attention_mask':batch.get('attention_mask')}
        outputs = self.Backbone(**inputs)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        # LSTM
        lstm_outputs,(lstm_hidden_state,lstm_cell_state) = self.lstm(sequence_output)
        #print(lstm_outputs.shape)
        # Average
        e1_h = self.entity_average(lstm_outputs, e1_mask)
        e2_h = self.entity_average(lstm_outputs, e2_mask)
        #print(e1_h.shape,e2_h.shape)
        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        return {'logits':logits}