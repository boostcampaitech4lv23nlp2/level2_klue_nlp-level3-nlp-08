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
    seed = 2022
    train_dataset = pd.read_csv("../NLP_dataset/train/train.csv")
    train_dataset = train_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)  
    print(train_dataset.head)
    train_label = label_to_num(train_dataset['label'].values)

    split_rate = 10
    all_label_counter = Counter(train_label)
    val_label_counter = {k: v//split_rate for k, v in all_label_counter.items()}

    train_out = []
    val_out = []
    for idx, row in tqdm(list(train_dataset.iterrows())):
        label = train_label[idx]
        if val_label_counter[label] > 0:
            val_out.append(row)
            val_label_counter[label] -= 1
        else:
            train_out.append(row)
            
    print(len(val_out), len(train_out))

    pd.DataFrame(train_out).to_csv(f"../NLP_dataset/train/train_{seed}_{split_rate}.csv", index=False)
    pd.DataFrame(val_out).to_csv(f"../NLP_dataset/train/val_{seed}_{split_rate}.csv", index=False)
