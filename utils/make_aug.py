import pandas as pd
from augmentation import *

def load_data(path):
    data = pd.read_csv(path)

    sub_entity, sub_type = [], []
    obj_entity, obj_type = [], []
    sub_idx, obj_idx = [], []
    sentence = []

    for idx, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
        subT = x[1:-1].split(':')[-1].split('\'')[-2] # Subject Entity의 type
        objT= y[1:-1].split(':')[-1].split('\'')[-2] # Object Entity의 type
        for idx_i in range(len(x)): # Entity label에서 start_idx와 end_idx 추출
            if x[idx_i:idx_i+9] == 'start_idx':
                sub_start = int(x[idx_i+12:].split(',')[0].strip())
            if x[idx_i:idx_i+7] == 'end_idx':
                sub_end = int(x[idx_i+10:].split(',')[0].strip())
                
            if y[idx_i:idx_i+9] == 'start_idx':
                obj_start = int(y[idx_i+12:].split(',')[0].strip())
            if y[idx_i:idx_i+7] == 'end_idx':
                obj_end = int(y[idx_i+10:].split(',')[0].strip())
      
        sub_i = [sub_start, sub_end]
        obj_i = [obj_start, obj_end]

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

def aeda(train_df, check_num):
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

        sentence_set = set([sentence])
        while True:
            new_sentence = make_new_text(sentence, punc_ratio)
            sentence_set.add(new_sentence)
            if len(sentence_set) >= check_num:
                break
        sentence_set.remove(sentence)

        for s in sentence_set:
            append_new_sentence(new_df, train_df, i, s)

    aug_df = train_df.append(new_df, ignore_index=True)
    pd.DataFrame(aug_df).to_csv(f"../NLP_dataset/train/train_aeda.csv", index=False)

if __name__ == '__main__':
    path = '../NLP_dataset/train/train.csv'
    dataset = load_data(path)

    aeda(dataset, 2)