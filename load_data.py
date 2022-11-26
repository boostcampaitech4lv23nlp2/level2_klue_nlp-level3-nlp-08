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
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
    
    return tokenized_sentences