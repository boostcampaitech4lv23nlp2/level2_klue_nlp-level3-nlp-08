import pickle as pickle
import os
import pandas as pd
import torch
from transformers import (
  AutoTokenizer,
  AutoConfig, 
  AutoModelForSequenceClassification, 
  Trainer, 
  TrainingArguments, 
  RobertaConfig, 
  RobertaTokenizer, 
  RobertaForSequenceClassification, 
  BertTokenizer,
  get_scheduler,
  EarlyStoppingCallback
)
from load_data import *
from utils.augmentation import *
import random
from utils.metric import *
from models import auto_models,R_BERT,R_BERT_BiLSTM,R_BERT_CNN,RoBERTa_BiLSTM
from trainer import *
import yaml
from omegaconf import OmegaConf
import argparse
import wandb
from transformers import logging

logging.set_verbosity_error()
def train():
  seed_fix() #Random seed fix

  MODEL_NAME = cfg.model.model_name #"klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  print('Data Loading...')
  train_preprocess = Preprocess(cfg.path.train_path)
  dev_preprocess = Preprocess(cfg.path.dev_path)

  train_dataset = train_preprocess.data
  dev_dataset = dev_preprocess.data

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  print('Data Tokenizing...')
  print(f'Selected Tokenize Type: {cfg.model.type}')
  if cfg.model.type == "rbert":
    tokenized_train,train_sub_ids,train_obj_ids = train_preprocess.tokenized_dataset(train_dataset, tokenizer,type=cfg.model.type,test=cfg.data.mode)
    tokenized_dev,dev_sub_ids,dev_obj_ids = dev_preprocess.tokenized_dataset(dev_dataset, tokenizer,type = cfg.model.type,test=cfg.data.mode)
    RE_train_dataset = RBERT_Dataset(tokenized_train, train_label,train_sub_ids,train_obj_ids)
    RE_dev_dataset = RBERT_Dataset(tokenized_dev, dev_label,dev_sub_ids,dev_obj_ids)
  else:
    tokenized_train = train_preprocess.tokenized_dataset(train_dataset, tokenizer,type=cfg.model.type,test=cfg.data.mode)
    tokenized_dev = dev_preprocess.tokenized_dataset(dev_dataset, tokenizer,type = cfg.model.type,test=cfg.data.mode)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(f'Selected Model Type: {cfg.model.type}')
  if cfg.model.type == "CNN":
    model = auto_models.CNN_Model(MODEL_NAME)
  elif cfg.model.type == "base":
    if cfg.model.type2 == "lstm":
      model = RoBERTa_BiLSTM.RoBERTa_BiLSTM(MODEL_NAME)
    else:
      model =  auto_models.RE_Model(MODEL_NAME)

  elif cfg.model.type == "entity":
    model = auto_models.EntityModel(MODEL_NAME)
  elif cfg.model.type == "rbert":
    if cfg.model.type2 == "lstm":
      model = R_BERT_BiLSTM.RBERT(MODEL_NAME)
    elif cfg.model.type2 == "cnn":
      model = R_BERT_CNN.RBERT(MODEL_NAME)
    else:
      model = R_BERT.RBERT(MODEL_NAME)

  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir= f'./results/{cfg.exp.exp_name}',          # output directory
    save_total_limit=cfg.train.save_total_limit, # number of total save model.
    save_steps=cfg.train.save_steps,                 # model saving step.
    num_train_epochs=cfg.train.max_epoch,              # total number of training epochs
    learning_rate=cfg.train.learning_rate,               # learning_rate
    per_device_train_batch_size= cfg.train.batch_size,  # batch size per device during training
    per_device_eval_batch_size= cfg.train.batch_size,   # batch size for evaluation
    warmup_steps=cfg.train.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay= cfg.train.weight_decay,               # strength of weight decay
    logging_dir='./logs/logs_BT_AEDA_1124',            # directory for storing logs
    logging_steps=cfg.train.logging_steps,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = cfg.train.eval_steps,            # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model= cfg.train.metric_for_best_model, #eval_loss
    greater_is_better = True,
    report_to='wandb',
    disable_tqdm = False
  )
  
  trainer = RE_Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,      # training dataset
    eval_dataset=RE_dev_dataset,       # evaluation dataset
    loss_name = cfg.train.loss_name,
    scheduler = cfg.train.scheduler,                   
    compute_metrics=compute_metrics,      # define metrics function
    num_training_steps = 3 * len(train_dataset),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.train.patience, early_stopping_threshold=0.0)],
    model_type = cfg.model.type
  )

  # train model
  wandb.watch(model)
  trainer.train()
  #try:
  #  model.save_pretrained(cfg.test.model_dir)
  #except:
  #  torch.save(model.state_dict(),cfg.test.model_dir)  

def main():
  wandb_cfg = dict()
  for root_key in cfg.keys():
      for key in cfg[root_key].keys():
        wandb_cfg[f'{root_key}.{key}'] = cfg[root_key][key]
  wandb.init(project = cfg.exp.project_name, name=cfg.exp.exp_name, entity='boot4-nlp-08', config=wandb_cfg)
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',type=str,default='base_config')
  args , _ = parser.parse_known_args()
  cfg = OmegaConf.load(f'./config/{args.config}.yaml')
  main()
