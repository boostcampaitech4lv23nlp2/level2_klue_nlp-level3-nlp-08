import pickle as pickle
import os
import pandas as pd
import torch
import tqdm


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

class Preprocess:
  def __init__(self, path, option):
    self.option = option
    self.data = self.load_data(path)
  
  def load_data(self, path):
    data = pd.read_csv(path)

    sub_entity, sub_type = [], []
    obj_entity, obj_type = [], []
    sub_idx, obj_idx = [], []
    sentence = []

    for idx, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
      subT = x[1:-1].split(':')[-1].split('\'')[-2] # Subject Entity의 type
      objT= y[1:-1].split(':')[-1].split('\'')[-2] # Object Entity의 type

      for idx_i in range(len(x)): # Entity label에서 start_idx와 end_idx 추출
        if x[idx_i:idx_i+9] == 'start_idx':
            sub_start = int(x[idx_i+12:].split(',')[0].strip())
        if x[idx_i:idx_i+7] == 'end_idx':
            sub_end = int(x[idx_i+10:].split(',')[0].strip())
                
        if y[idx_i:idx_i+9] == 'start_idx':
            obj_start = int(y[idx_i+12:].split(',')[0].strip())
        if y[idx_i:idx_i+7] == 'end_idx':
            obj_end = int(y[idx_i+10:].split(',')[0].strip())
      
      sub_i = [sub_start, sub_end]
      obj_i = [obj_start, obj_end]

      sub_entity.append(z[sub_i[0]:sub_i[1]+1])
      obj_entity.append(z[obj_i[0]:obj_i[1]+1])
      sub_type.append(subT)
      sub_idx.append(sub_i)
      obj_type.append(objT)
      obj_idx.append(obj_i)

      if self.option == 'TOKEN': # Sub/Obj Entity 양 옆에 [SUB][/SUB] or [OBJ][/OBJ] 토큰 삽입
        if sub_i[0] < obj_i[0]:
            z = z[:sub_i[0]] + '[SUB]' + z[sub_i[0]:sub_i[1]+1] + '[/SUB]' + z[sub_i[1]+1:]
            z = z[:obj_i[0]+11] + '[OBJ]' + z[obj_i[0]+11: obj_i[1]+12] + '[/OBJ]'+ z[obj_i[1]+12:]
        else:
            z = z[:obj_i[0]] + '[OBJ]' + z[obj_i[0]: obj_i[1]+1] + '[/OBJ]'+ z[obj_i[1]+1:]
            z = z[:sub_i[0]+11] + '[SUB]'+ z[sub_i[0]+11: sub_i[1]+12] + '[/SUB]' + z[sub_i[1]+12:]

      elif self.option == 'PUNCT': # Sub/Obj Entity 양 옆에 @/# 기호 삽입, 추가로 entity 왼쪽에 *entity type*/^entity type^삽입
          if sub_i[0] < obj_i[0]:
              z = z[:sub_i[0]] + '@*' + subT +'*' + z[sub_i[0]: sub_i[1]+1] + '@' + z[sub_i[1]+1:]
              z = z[:obj_i[0]+7] + '#^' + objT + '^'+ z[obj_i[0]+7: obj_i[1]+8] + '#'+ z[obj_i[1]+8:]
          else:
              z = z[:obj_i[0]] + '#^' + objT +'^' + z[obj_i[0]: obj_i[1]+1] + '#' + z[obj_i[1]+1:]
              z = z[:sub_i[0]+7] + '@*' + subT + '*' + z[sub_i[0]+7: sub_i[1]+8] + '@' + z[sub_i[1]+8:]
      
      sentence.append(z)
    
    df = pd.DataFrame({'id': data['id'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
                       'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
                       'subject_idx': sub_idx, 'object_idx': obj_idx})
    
    return df
  
  def label_to_num(self, label):
    num_label = []

    with open('./NLP_dataset/dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num= pickle.load(f)
      for val in label:
          num_label.append(dict_label_to_num[val])
        
    return num_label
  
  def tokenized_dataset(self, dataset, tokenizer):

    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    entity_ids = []
    for e01, e02, sent in tqdm.tqdm(zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']), total=len(dataset)):
        temp = ''
        temp = e01 + '[SEP]' + e02
        entity_one_ids = tokenizer(e01, add_special_tokens=False)['input_ids']
        entity_two_ids = tokenizer(e02, add_special_tokens=False)['input_ids']
        tokenized_sent = tokenizer(temp, sent, padding="max_length", truncation=True, max_length=256, add_special_tokens=True)
        tokenized_input_ids = tokenized_sent['input_ids']
        search_length = min(len(entity_one_ids), len(entity_two_ids))
        entity_temp = [0] * 256

        for i in range(len(tokenized_input_ids) - (search_length-2)):
        
            if tokenized_input_ids[i: i+len(entity_one_ids)-2] == entity_one_ids[1:-1]:
                entity_temp[i: i+len(entity_one_ids)-2] = [1] * (len(entity_one_ids) - 2)

            elif tokenized_input_ids[i: i+len(entity_two_ids)-2] == entity_two_ids[1:-1]:
                entity_temp[i: i+len(entity_two_ids)-2] = [1] * (len(entity_two_ids) - 2)

        entity_ids.append(entity_temp)

        concat_entity.append(temp)
    
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
    
    tokenized_sentences['entity_ids'] = torch.LongTensor(entity_ids)

    return tokenized_sentences