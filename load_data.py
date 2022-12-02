import pickle as pickle
import os
import pandas as pd
import torch
import tqdm
from utils import make_entity_ids
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

class RBERT_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels,sub_ids,obj_ids):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.sub_ids = sub_ids
    self.obj_ids = obj_ids

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['sub_ids'] = torch.tensor(self.sub_ids[idx])
    item['obj_ids'] = torch.tensor(self.obj_ids[idx])
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
  
  def tokenized_dataset(self, dataset, tokenizer,type=False,test=False):
    if type == 'rbert':
      sub_list = []
      obj_list = []

      for sent in dataset['sentence']:
        sub_id,obj_id = make_entity_ids.make_ent_ids(tokenizer,sent)
        sub_list.append(sub_id)
        obj_list.append(obj_id)
      if test:
        tmp = []
        for e01,e02,e03,e04 in zip(dataset['subject_entity'],dataset['object_entity'],dataset['subject_type'],dataset['object_type']):
          ex = f"@*{e03}*{e01}@ 와(과) #^{e04}^{e02}# 의 관계"
          tmp.append(ex)
        tokenized_sentences = tokenizer(
            tmp,
            list(dataset['sentence']),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            )
      else:
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,#160
            add_special_tokens=True,
            )
      return tokenized_sentences,sub_list,obj_list

    elif type == 'entity':
      print(dataset['sentence'].iloc[0:10])

      entity_loc_ids = []
      entity_type_ids = []

      for sent, sub_type, obj_type in zip(dataset['sentence'], dataset['subject_type'], dataset['object_type']):
        current_entity_loc_ids, current_entity_type_ids = make_entity_ids.make_entity_ids(sentence=sent, tokenizer=tokenizer)
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
      
    elif type == 'xlm':
      concat_entity = []
      for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
      
      tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
      return tokenized_sentences

    else:
      if test:
        tmp = []
        for e01,e02,e03,e04 in zip(dataset['subject_entity'],dataset['object_entity'],dataset['subject_type'],dataset['object_type']):
          ex = f"@*{e03}*{e01}@ 와(과) #^{e04}^{e02}# 의 관계"
          tmp.append(ex)
        tokenized_sentences = tokenizer(
            tmp,
            list(dataset['sentence']),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            )
      else:
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            )
    
      return tokenized_sentences