from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
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
from models import *
import datetime
from utils.metric import label_to_num
from pytz import timezone


def inference(model, tokenized_sent, device):
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
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    if cfg.model.type == 'CNN':
      logits = outputs.get('logits')
    elif cfg.model.type == 'base':
      logits = outputs[0]
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
  tokenized_test = dataset.tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(cfg):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = cfg.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load my model
  MODEL_NAME = cfg.model.model_name # model dir.
  if cfg.model.type == 'base':
    model = auto_models.RE_Model(MODEL_NAME)
  elif cfg.model.type == 'CNN':
    model = auto_models.CNN_Model(MODEL_NAME)
  elif cfg.model.type == 'enitity':
    if cfg.model.model_name == "klue/bert-base":
      config = AutoConfig.from_pretrained(MODEL_NAME)
      model = custom_model.BertForSequenceClassification(config)
    elif cfg.model.model_name == "monologg/koelectra-base-v3-discriminator":
      config = AutoConfig.from_pretrained(MODEL_NAME)
      model = custom_model.ElectraForSequenceClassification(config)

  if isinstance(model, PreTrainedModel):
    model = model.from_pretrained('checkpoint', num_labels=30)
  else:
    best_state_dict= torch.load(cfg.test.model_dir)
    model.load_state_dict(best_state_dict)
    
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = cfg.path.predict_path
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
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
  tokenized_dev = val_process.tokenized_dataset(dev_dataset, tokenizer)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  
  _, output_prob = inference(model, RE_dev_dataset, device) # model에서 class 추론
  result = [' '.join(map(lambda x: f'{x:.3f}', out)) for out in output_prob]
  dev_dataset['output_prob'] = result
  time = get_time()
  dev_dataset.to_csv(f"EDA/output/{cfg.exp.exp_name}_{time}.csv", index=False)
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
  
