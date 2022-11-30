from collections import defaultdict
from torch import nn
from transformers import Trainer, get_scheduler, TrainingArguments
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from typing import Iterator, Sized
import pandas as pd
import numpy as np
import random
from loss import *
from pytorch_metric_learning.losses import NTXentLoss
from torchsampler import ImbalancedDatasetSampler


class RE_Trainer(Trainer):
    def __init__(self, loss_name, 
                       scheduler,
                       num_training_steps,model_type,
                       *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name
        self.scheduler = scheduler
        self.num_training_steps = num_training_steps
        self.model_type = model_type

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # forward pass
        if self.model_type == 'CNN':
          inputs = {'input_ids':inputs.get('input_ids'),'token_type_ids':inputs.get('token_type_ids'),'attention_mask':inputs.get('attention_mask')}
          outputs = model(**inputs)
          logits = outputs.get("logits")

        elif self.model_type == 'base':
            inputs = {'input_ids':inputs.get('input_ids'),'token_type_ids':inputs.get('token_type_ids'),'attention_mask':inputs.get('attention_mask'),'labels':inputs.get('labels')}
            outputs = model(**inputs)
            logits = outputs.get("logits")

        elif self.model_type == 'entity':
            outputs = model(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'],
                        attention_mask=inputs['attention_mask'], entity_ids=inputs['entity_ids'])
            logits = outputs['outputs']
            
        elif self.model_type == 'CT':
            inputs = {'input_ids':inputs.get('input_ids'),'token_type_ids':inputs.get('token_type_ids'),'attention_mask':inputs.get('attention_mask')}
            outputs = model(**inputs)
            logits = outputs.get("logits")
            embedding = outputs.get("embedding")
            
        elif self.model_type == 'DAE_CT':
            inputs = {'input_ids':inputs.get('input_ids'),'token_type_ids':inputs.get('token_type_ids'),'attention_mask':inputs.get('attention_mask')}
            outputs = model(**inputs)
            logits = outputs.get("logits")
            embedding = outputs.get("embedding") 
            ori_embedding = outputs.get("ori_embedding") # pre-trained embedding

        # compute custom loss (suppose one has 3 labels with different weights)
        if self.loss_name == 'CE':
          loss_fct = nn.CrossEntropyLoss()
        elif self.loss_name == 'LBS':
          loss_fct = LabelSmoothingLoss()
        elif self.loss_name == 'focal':
          loss_fct = FocalLoss()
        elif self.loss_name == 'f1':
          loss_fct = F1Loss()
        
        if self.model_type == 'CT':
          loss = loss_fct(logits.view(-1, 30), labels.view(-1))
          ntxent_loss = NTXentLoss(temperature=0.5)
          loss = 0.25*loss + 0.75*ntxent_loss(embedding, labels)
          
        elif self.model_type == 'DAE_CT':
          loss = loss_fct(logits.view(-1, 30), labels.view(-1))
          ntxent_loss = NTXentLoss(temperature=0.5)
          bottleneck_loss = torch.nn.CosineEmbeddingLoss()
          target = torch.Tensor(embedding.size(0)).cuda().fill_(1.0)
          #loss = 0.8*loss + 0.2*bottleneck_loss(embedding, ori_embedding, target)
          #loss = 0.8*loss + 0.2*ntxent_loss(embedding, labels)
        else:
          loss = loss_fct(logits.view(-1, 30), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def create_scheduler(self, num_training_steps, optimizer= None ):
      if self.scheduler == 'linear' or self.scheduler == 'cosine':
        if self.scheduler == 'linear':
          my_scheduler = "linear"
        elif self.scheduler == 'cosine':
          my_scheduler = "cosine_with_restarts"

        self.lr_scheduler = get_scheduler(
            my_scheduler,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
          )

      elif self.scheduler == 'steplr':
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1080, gamma=0.5)

      return self.lr_scheduler


class ImbalancedSamplerTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        def get_label(dataset):
            return dataset["labels"]

        train_sampler = ImbalancedDatasetSampler(
            train_dataset, callback_get_label=get_label
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
class ContrastiveSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: Sized, batch_size: int) -> None:
        print(batch_size)
        reserved_num = 2
        self.data_source = data_source
        self.indices = list(range(len(self.data_source)))
        df = pd.DataFrame() 
        df['label'] = self.data_source.labels
        label_arr = df['label'].unique()  # unique 라벨 리스트.
        label_num = len(label_arr) # 라벨 개수
        res_indices_list = [] # residual 라벨 index list
        indices_dict = {label: df[df['label'] == label].index for label in label_arr}
        label_pos = {label: 0 for label in label_arr} # 각 라벨들의 index 위치
        self.new_indices = []
        label_index = 0 # similar label 순회를 위한 index
        sample_index = 0 # 샘플 순회를 위한 index
        loop = True
        while loop:
          batch = []
          label = label_arr[label_index]
          # same class sample
          for rev_idx in range(reserved_num):
            pos = (label_pos[label] + rev_idx) % len(indices_dict[label])
            batch.append(indices_dict[label][pos])
          label_pos[label] = (label_pos[label] + reserved_num) % len(indices_dict[label])
          label_index = (label_index + 1) % label_num
          # 이전 batch에서 중복된 label 합침
          while res_indices_list and len(batch) < batch_size:
            if random.random() + 0.1 < 1/len(res_indices_list) :
                break
            if True: #df['label'][res_indices_list[-1]] != label:
              batch.append(res_indices_list.pop())
              
              
          # batch 남은 샘플 채우기
          while len(batch) < batch_size:
            if False: # df['label'][sample_index] == label:
              res_indices_list.append(sample_index)
            else:
              batch.append(sample_index)
            if sample_index >= len(self.indices) - 1:
              sample_index = 0
              loop = False
            sample_index = (sample_index + 1) % len(self.indices)
            
          self.new_indices.extend(batch)
          
          
    def __iter__(self) -> Iterator[int]:
        return iter(self.new_indices)

    def __len__(self) -> int:
        return len(self.new_indices)

    
    
class CT_Trainer(RE_Trainer):
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        def get_label(dataset):
            return dataset["label"]

        train_sampler = ContrastiveSampler(train_dataset, batch_size=self.args.per_device_train_batch_size)


        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
