import pandas as pd
from augmentation import *
import argparse

def load_data(args, path):
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
        else:
            sub_i = [sub_start, sub_end]
            obj_i = [obj_start-1, obj_end-1]

        sub_entity.append(z[sub_i[0]:sub_i[1]+1])
        obj_entity.append(z[obj_i[0]:obj_i[1]+1])
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

def aeda(args, train_df, check_num):
    train_df = train_df.reset_index(drop=True)

    new_df = pd.DataFrame(
        [], columns=["id", "sentence", "subject_entity", "object_entity", "subject_type", "object_type", 
                     "label", "subject_idx", "object_idx"]
    )
    new_label = []

    for i in tqdm(range(len(train_df)), desc="Data Augmentation Processing..."):
        sentence = train_df.iloc[i]["sentence"]

        if len(sentence) <= 200:
            punc_ratio = 0.2
        elif len(sentence) <= 300:
            punc_ratio = 0.3
        else:
            punc_ratio = 0.35

        subject_entity = "@" + sentence.split("@")[1] + "@"
        object_entity = "#" + sentence.split("#")[1] + "#"
        
        sentence_set = set([sentence])
        while True:
            new_sentence = make_new_text(sentence, punc_ratio, subject_entity, object_entity)
            sentence_set.add(new_sentence)
            if len(sentence_set) >= check_num:
                break
        sentence_set.remove(sentence)

        for s in sentence_set:
            append_new_sentence(new_df, train_df, i, s)

    aug_df = train_df.append(new_df, ignore_index=True)
    pd.DataFrame(aug_df).to_csv(f"../NLP_dataset/train/{args.train_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default='train')
    parser.add_argument('--train_file', type=str, default='BT_AEDA_train.csv')
    parser.add_argument('--test_file', type=str, default='BT_test.csv')
    args = parser.parse_args()
    
    if args.option == 'train':
        path = '../NLP_dataset/train/train.csv'
        dataset = load_data(args, path)
        aeda(args, dataset, 2)
    elif args.option == 'test':
        path = '../NLP_dataset/test/test_data.csv'
        dataset = load_data(args, path)
        pd.DataFrame(dataset).to_csv(f'../NLP_dataset/test/{args.test_file}')