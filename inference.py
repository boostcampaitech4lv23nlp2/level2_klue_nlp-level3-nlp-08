from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, logging
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from models import auto_models,R_BERT,R_BERT_BiLSTM,R_BERT_CNN,RoBERTa_BiLSTM,custom_embedding,custom_model
import datetime
from utils.metric import label_to_num
from pytz import timezone

logging.set_verbosity_error()
def inference(cfg, model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(dataloader):  # tqdm
    with torch.no_grad():
      if cfg.model.type == 'xlm':
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          )
      elif cfg.model.type == 'rbert':
        outputs = model (data['sub_ids'].to(device),data['obj_ids'].to(device),
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
      elif cfg.model.type == 'entity':
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device),
          entity_loc_ids=data['entity_loc_ids'].to(device)
                    )
      elif cfg.model.type == 'type':
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device),
          entity_loc_ids=data['entity_loc_ids'].to(device),
          entity_type_ids=data['entity_type_ids'].to(device)
                    )
      elif cfg.model.type == 'specific':
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device),
          entity_loc_ids=data['entity_loc_ids'].to(device)
                    )
      else:
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device)
            )

      if cfg.model.type == 'CNN':
        logits = outputs.get('logits')
      elif cfg.model.type == 'base' or cfg.model.type == 'xlm':
        logits = outputs[0]
      elif cfg.model.type == 'base':
        logits = outputs.get('logits')
      elif cfg.model.type == 'rbert':
        logits = outputs.get('logits')
      elif cfg.model.type == 'entity':
        logits = outputs['logits']
      elif cfg.model.type == 'type':
        logits = outputs['logits']
      elif cfg.model.type == 'specific':
        logits = outputs['logits']

    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  dataset = Preprocess(dataset_dir)
  test_dataset = dataset.load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  if cfg.model.type == 'rbert':
    tokenized_test,sub_list,obj_list = dataset.tokenized_dataset(test_dataset, tokenizer, cfg.model.type,cfg.data.mode)
    return test_dataset['id'], tokenized_test,sub_list,obj_list, test_label
  else:
    tokenized_test = dataset.tokenized_dataset(test_dataset, tokenizer, cfg.model.type,cfg.data.mode)
    return test_dataset['id'], tokenized_test, test_label

def main(cfg):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #'cuda:0' if torch.cuda.is_available() else 
  # load tokenizer
  Tokenizer_NAME = cfg.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load my model
  MODEL_NAME = cfg.model.model_name # model dir.
  if cfg.model.type == 'base':
    if cfg.model.type2 == "lstm":
      model = RoBERTa_BiLSTM.RoBERTa_BiLSTM(MODEL_NAME)
    else:
      model =  auto_models.RE_Model(MODEL_NAME)
  elif cfg.model.type == 'CNN':
    model = auto_models.CNN_Model(MODEL_NAME)
  elif cfg.model.type =='specific':
    model = auto_models.SpecificModel(MODEL_NAME)
  elif cfg.model.type == 'enitity' or cfg.model.type == 'type':
    if cfg.model.model_name == "klue/bert-base":
      config = AutoConfig.from_pretrained(MODEL_NAME)
      model = custom_model.BertForSequenceClassification(config)
    elif cfg.model.model_name == "monologg/koelectra-base-v3-discriminator":
      config = AutoConfig.from_pretrained(MODEL_NAME)
      model = custom_model.ElectraForSequenceClassification(config)
    elif cfg.model.model_name == "klue/roberta-large":
      config = AutoConfig.from_pretrained(MODEL_NAME)
      model = custom_model.RobertaForSequenceClassification(config)
  elif cfg.model.type == 'xlm':
    model = auto_models.RE_Model(MODEL_NAME)
  elif cfg.model.type =='rbert':
    if cfg.model.type2 == 'lstm':
      model = R_BERT_BiLSTM.RBERT(MODEL_NAME)
    elif cfg.model.type2 == 'cnn':
      model = R_BERT_CNN.RBERT(MODEL_NAME)
    else:
      model = R_BERT.RBERT(MODEL_NAME)

  if isinstance(model, PreTrainedModel):
    model = model.from_pretrained('checkpoint', num_labels=30)
  else:
    best_state_dict= torch.load(cfg.test.model_dir)
    model.load_state_dict(best_state_dict)

  
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = cfg.path.predict_path
  if cfg.model.type == 'rbert':
    test_id, test_dataset,sub_list,obj_list,test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RBERT_Dataset(test_dataset,test_label,sub_list,obj_list)

  else:
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(cfg, model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv(cfg.test.prediction, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
  
  val_process = Preprocess(cfg.path.dev_path)
  dev_dataset = val_process.data
  dev_label = label_to_num(dev_dataset['label'].values)
  if cfg.model.type == 'rbert':
    tokenized_dev, sub_mask, obj_mask = val_process.tokenized_dataset(dev_dataset, tokenizer, cfg.model.type , cfg.data.mode)
    RE_dev_dataset = RBERT_Dataset(tokenized_dev, dev_label, sub_mask, obj_mask)
  else:
    tokenized_dev = val_process.tokenized_dataset(dev_dataset, tokenizer)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  
  _, output_prob = inference(cfg, model, RE_dev_dataset, device) # model에서 class 추론
  result = [' '.join(map(lambda x: f'{x:.3f}', out)) for out in output_prob]
  dev_dataset['output_prob'] = result
  time = get_time()
  dev_dataset.to_csv(f"./EDA/output/{cfg.exp.exp_name}_{time}.csv", index=False)
  print('----csv generate Finish! ----')
  
def get_time():
    now = str(datetime.datetime.now(timezone('Asia/Seoul')))
    date, time = now.split(" ")
    y, m, d = date.split("-")
    time = time.split(".")[0]
    return y[2:]+m+d+"-"+time




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',type=str,default='')
  args , _ = parser.parse_known_args()
  cfg = OmegaConf.load(f'./config/{args.config}.yaml')
  
  # model dir
  main(cfg)
  
