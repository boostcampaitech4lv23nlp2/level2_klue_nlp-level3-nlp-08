import pandas as pd


train = pd.read_csv('../NLP_dataset/train/train.csv')
test = pd.read_csv('../NLP_dataset/test/test_data.csv')

train_data = train.iloc[0]

print(f"Sentence: {train_data['sentence']}")
print(f"Subject_entity: {train_data['subject_entity']}")
print(f"Object_entity: {train_data['object_entity']}")
print(f"Label: {train_data['label']}")
print(f"Source: {train_data['source']}")

print(train.shape)
print(test.shape)

print(train['label'].unique())