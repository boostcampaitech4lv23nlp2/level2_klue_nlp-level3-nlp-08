import pickle as pickle
import os
import pandas as pd
import torch
import tqdm
import numpy as np


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
  def __init__(self, path):
    self.data = self.load_data(path)
  
  def load_data(self, path):
    data = pd.read_csv(path)
    
    return data
  
  def label_to_num(self, label):
    num_label = []

    with open('./NLP_dataset/dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num= pickle.load(f)
      for val in label:
          num_label.append(dict_label_to_num[val])
        
    return num_label
  
  def tokenized_dataset(self, dataset, tokenizer):
    print(dataset['sentence'].iloc[0:10])

    entity_loc_ids = []
    entity_type_ids = []

    for sent, sub_type, obj_type in zip(dataset['sentence'], dataset['subject_type'], dataset['object_type']):
      current_entity_loc_ids, current_entity_type_ids = self.make_entity_ids(sentence=sent, tokenizer=tokenizer)
      entity_loc_ids.append(current_entity_loc_ids)
      entity_type_ids.append(current_entity_type_ids)

    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
        
    tokenized_sentences['entity_loc_ids'] = torch.LongTensor(entity_loc_ids)
    tokenized_sentences['entity_type_ids'] = torch.LongTensor(entity_type_ids)
    return tokenized_sentences

  def make_entity_ids(self, sentence, tokenizer):

    entity_loc_ids = [0] * 256
    entity_type_ids = [0] * 256

    type_to_num={
        '사람': 1,
        '조직': 2,
        '날짜': 3,
        '장소': 4,
        '단어': 5,
        '숫자': 6,
      }

    tokenized_sentence = tokenizer.tokenize(sentence, padding="max_length", truncation=True, max_length=256, add_special_tokens=True)
    tokenized_sentence = np.array(tokenized_sentence)
    sub_indices = np.where(tokenized_sentence == '@')[0]
    sub_type_indices = np.where(tokenized_sentence == '*')[0]
    obj_indices = np.where(tokenized_sentence == '#')[0]
    obj_type_indices = np.where(tokenized_sentence == '^')[0]

    entity_loc_ids[sub_type_indices[-1]+1: sub_indices[-1]] = [1] * (sub_indices[-1] - sub_type_indices[-1]-1)
    entity_loc_ids[obj_type_indices[-1]+1: obj_indices[-1]] = [2] * (obj_indices[-1] - obj_type_indices[-1]-1) 

    entity_type_ids[sub_type_indices[0]+1] = type_to_num[tokenized_sentence[sub_type_indices[0]+1]]
    entity_type_ids[obj_type_indices[0]+1] = type_to_num[tokenized_sentence[obj_type_indices[0]+1]]
            
    return entity_loc_ids, entity_type_ids