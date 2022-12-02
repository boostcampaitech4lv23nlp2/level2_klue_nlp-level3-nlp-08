import numpy as np

def make_ent_ids(tokenizer,data):
    sub_ids = [0]*256
    obj_ids = [0]*256

    tokens = tokenizer.tokenize(data)
    tokens = np.array(tokens)

    sub_idx1 = np.where(tokens=="@")[0]
    sub_idx2 = np.where(tokens=="*")[0]

    obj_idx1 = np.where(tokens=="#")[0]
    obj_idx2 = np.where(tokens=="^")[0]
    
    for i in range(sub_idx2[1],sub_idx1[1]):
        sub_ids[i]=1
    
    for j in range(obj_idx2[1],obj_idx1[1]):
        obj_ids[j]=2

    return sub_ids,obj_ids

def make_entity_ids(self, sentence, tokenizer):

    entity_loc_ids = [0] * 256
    entity_type_ids = [0] * 256

    type_to_num={
        '사람': 1,
        '조직': 2,
        '날짜': 3,
        '장소': 4,
        '단어': 5,
        '숫자': 6,
      }

    tokenized_sentence = tokenizer.tokenize(sentence, padding="max_length", truncation=True, max_length=256, add_special_tokens=True)
    tokenized_sentence = np.array(tokenized_sentence)
    sub_indices = np.where(tokenized_sentence == '@')[0]
    sub_type_indices = np.where(tokenized_sentence == '*')[0]
    obj_indices = np.where(tokenized_sentence == '#')[0]
    obj_type_indices = np.where(tokenized_sentence == '^')[0]

    entity_loc_ids[sub_type_indices[-1]+1: sub_indices[-1]] = [1] * (sub_indices[-1] - sub_type_indices[-1]-1)
    entity_loc_ids[obj_type_indices[-1]+1: obj_indices[-1]] = [2] * (obj_indices[-1] - obj_type_indices[-1]-1) 

    entity_type_ids[sub_type_indices[0]+1] = type_to_num[tokenized_sentence[sub_type_indices[0]+1]]
    entity_type_ids[obj_type_indices[0]+1] = type_to_num[tokenized_sentence[obj_type_indices[0]+1]]
            
    return entity_loc_ids, entity_type_ids