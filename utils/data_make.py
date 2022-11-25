import pandas as pd
import pickle
from collections import Counter
from tqdm import tqdm

def label_to_num(label):
  num_label = []
  with open('../NLP_dataset/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

if __name__ == '__main__':   
    seed = 42
    train_dataset = pd.read_csv("../NLP_dataset/train/BT_train.csv")
    train_dataset = train_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)  
    train_label = label_to_num(train_dataset['label'].values)

    split_rate = 10
    all_label_counter = Counter(train_label)
    val_label_counter = {k: v//split_rate for k, v in all_label_counter.items()}

    train_out = []
    val_out = []
    val_id = []
    for idx, row in tqdm(list(train_dataset.iterrows())):
        label = train_label[idx]
        row_id = row['id']
        if val_label_counter[label] > 0 and row_id not in val_id:
            val_out.append(row)
            val_label_counter[label] -= 1
            val_id.append(row_id)
        else:
            train_out.append(row)
            
    print(len(val_out), len(train_out))

    pd.DataFrame(train_out).to_csv(f"../NLP_dataset/train/BT_split_train.csv", index=False)
    pd.DataFrame(val_out).to_csv(f"../NLP_dataset/train/BT_split_valid.csv", index=False)
