import pandas as pd
train = pd.read_csv('../NLP_dataset/train/BT_AEDA_train_final.csv')
val = pd.read_csv('../NLP_dataset/train/final_valid_data.csv')

print(any(train['id'].isin(val['id'])))  # 검증
filter = train['id'].isin(val['id'])
train = train[~filter]
train.to_csv('../NLP_dataset/train/final_train_data.csv')
print(any(train['id'].isin(val['id'])))  # 검증