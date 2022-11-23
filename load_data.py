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

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  entity_ids = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
    
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding='max_length',
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def entity_tokenized_dataset(dataset, tokenizer):

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