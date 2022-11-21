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
  get_scheduler
)
from load_data import *
from utils.metric import *
from models import *
from trainer import *
import yaml
from omegaconf import OmegaConf
import argparse
import wandb


def train():
  # fixed seed
  seed_fix()
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"

  MODEL_NAME = cfg.model.model_name #"klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data(cfg.path.train_path)
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  
  model =  auto_models.RE_Model(MODEL_NAME)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=cfg.train.save_total_limit,# number of total save model.
    save_steps=cfg.train.save_steps,                 # model saving step.
    num_train_epochs=cfg.train.max_epoch,              # total number of training epochs
    learning_rate=cfg.train.learning_rate,               # learning_rate
    per_device_train_batch_size= cfg.train.batch_size,  # batch size per device during training
    per_device_eval_batch_size= cfg.train.batch_size,   # batch size for evaluation
    warmup_steps=cfg.train.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay= cfg.train.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=cfg.train.logging_steps,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = cfg.train.eval_steps,            # evaluation step.
    load_best_model_at_end = True,
    
  )
  
  trainer = RE_Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,      # training dataset
    eval_dataset=RE_train_dataset,       # evaluation dataset
    loss_name = cfg.train.loss_name,
    scheduler = cfg.train.scheduler,                   
    compute_metrics=compute_metrics,      # define metrics function
    num_training_steps = 3 * len(train_dataset)
  )

  # train model
  wandb.watch(model)
  trainer.train()
  torch.save(model.state_dict(),cfg.test.model_dir)  

def main():
  wandb.init(project = 'lhJoon_exp',name=cfg.exp.exp_name,entity='boot4-nlp-08')
  wandb.config = cfg
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',type=str,default='')
  args , _ = parser.parse_known_args()
  cfg = OmegaConf.load(f'./config/{args.config}.yaml')
  main()
