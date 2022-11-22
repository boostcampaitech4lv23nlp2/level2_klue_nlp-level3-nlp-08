import pickle as pickle
import os
import pandas as pd
import torch
import re

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
    print(i)
    breakpoint()
    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def ner_preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  ss_entity=[]
  se_entity=[]
  os_entity=[]
  oe_entity=[]
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    sepi=i.split('start_idx')
    sepj=j.split('start_idx')
    ss,se=re.findall(r'[0-9]+',sepi[1])
    os,oe=re.findall(r'[0-9]+',sepj[1])
    i = i.split('\'type\': ')
    j = j.split('\'type\': ')
    i=re.search(r'[A-z]+',i[1]).group()
    j=re.search(r'[A-z]+',j[1]).group()

    subject_entity.append(i)
    object_entity.append(j)
    ss_entity.append(ss)
    se_entity.append(se)
    os_entity.append(os)
    oe_entity.append(oe)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],'ss':ss_entity,'se':se_entity,'os':os_entity,'oe':oe_entity,})
  
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = ner_preprocessing_dataset(pd_dataset)
  
  return dataset


def tokenized_dataset(dataset, tokenizer,mode=None):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""

  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  
  if mode=='typed_entity_marker':
     new_tokens=[]
     for sen,e01,e02, ss,se,os,oe in zip(dataset['sentence'], dataset['subject_entity'],dataset['object_entity'],dataset['ss'],dataset['se'],dataset['os'],dataset['oe']):
        subj_start = '[SUBJ-{}]'.format(e01)
        subj_end='[/SUBJ-{}]'.format(e01)
        obj_start='[OBJ-{}]'.format(obj_type)
        obj_end='[/OBJ-{}]'.format(obj_type)
        for token in (subj_start,subj_end,obj_start,obj_end):
          if token not in new_tokens:
            print(token)    
            new_tokens.append(token)
            tokenizer.add_tokens([token])
        
        breakpoint()
        if ss<=os:
          temp=sen[0:ss]+subj_start+sen[ss:se+1]+subj_end+sen[se+1:os]+obj_start+sen[os:oe+1]+obj_end+sen[oe+1:]
        else:
          temp=sen[0:os]+obj_start+sen[os:oe+1]+obj_end+sen[oe+1:ss]+subj_start+sen[ss:se+1]+subj_end+sen[se+1:]
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

