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