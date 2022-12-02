import pandas as pd
from augmentation import *
import argparse

def replace_token(string):
    string = string.replace('#', '')
    string = string.replace('@', '')
    string = string.replace('^', '')
    string = string.replace('*', '')

    return string

def load_data(path):
    data = pd.read_csv(path)

    sub_entity, sub_type = [], []
    obj_entity, obj_type = [], []
    sub_idx, obj_idx = [], []
    sentence = []
    match_dict={
        'PER': '사람',
        'ORG': '조직',
        'DAT': '날짜',
        'LOC': '장소',
        'POH': '단어',
        'NOH': '숫자',
    }

    for idx, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
        x, y, z = replace_token(x), replace_token(y), replace_token(z)
        subT = x[1:-1].split(':')[-1].split('\'')[-2] # Subject Entity의 type
        objT= y[1:-1].split(':')[-1].split('\'')[-2] # Object Entity의 type
        subT, objT = match_dict[subT], match_dict[objT]
        maxlen = max(len(x), len(y))

        for idx_i in range(maxlen): # Entity label에서 start_idx와 end_idx 추출
            if x[idx_i:idx_i+9] == 'start_idx':
                sub_start = int(x[idx_i+12:].split(',')[0].strip())
            if x[idx_i:idx_i+7] == 'end_idx':
                sub_end = int(x[idx_i+10:].split(',')[0].strip())
                
            if y[idx_i:idx_i+9] == 'start_idx':
                obj_start = int(y[idx_i+12:].split(',')[0].strip())
            if y[idx_i:idx_i+7] == 'end_idx':
                obj_end = int(y[idx_i+10:].split(',')[0].strip())

        if sub_start > obj_start:
            sub_i = [sub_start-1, sub_end-1]
            obj_i = [obj_start, obj_end]
            sub_entity.append(z[sub_i[0]+1:sub_i[1]+2])
            obj_entity.append(z[obj_i[0]:obj_i[1]+1])
        else:
            sub_i = [sub_start, sub_end]
            obj_i = [obj_start-1, obj_end-1]
            sub_entity.append(z[sub_i[0]:sub_i[1]+1])
            obj_entity.append(z[obj_i[0]+1:obj_i[1]+2])

        
        sub_type.append(subT)
        sub_idx.append(sub_i)
        obj_type.append(objT)
        obj_idx.append(obj_i)

        if sub_i[0] < obj_i[0]:
            z = z[:sub_i[0]] + '@*' + subT +'*' + z[sub_i[0]: sub_i[1]+1] + '@' + z[sub_i[1]+1:]
            z = z[:obj_i[0]+7] + '#^' + objT + '^'+ z[obj_i[0]+7: obj_i[1]+8] + '#'+ z[obj_i[1]+8:]
        else:
            z = z[:obj_i[0]] + '#^' + objT +'^' + z[obj_i[0]: obj_i[1]+1] + '#' + z[obj_i[1]+1:]
            z = z[:sub_i[0]+7] + '@*' + subT + '*' + z[sub_i[0]+7: sub_i[1]+8] + '@' + z[sub_i[1]+8:]
      
        sentence.append(z)
    
    df = pd.DataFrame({'id': data['id'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
                       'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
                       'subject_idx': sub_idx, 'object_idx': obj_idx})
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train_path = '../NLP_dataset/train/BT_split_train_final.csv'
    valid_path = '../NLP_dataset/train/BT_split_valid_final.csv'
    test_path = '../NLP_dataset/test/test_data.csv'

    train_preprocess = load_data(train_path)
    valid_preprocess = load_data(valid_path)
    test_preprocess = load_data(test_path)
    pd.DataFrame(train_preprocess).to_csv(f'../NLP_dataset/train/BT_train_preprocess_final.csv')
    pd.DataFrame(valid_preprocess).to_csv(f'../NLP_dataset/train/final_valid_data.csv')
    pd.DataFrame(test_preprocess).to_csv(f'../NLP_dataset/test/final_test_data.csv')
    
