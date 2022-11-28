import pandas as pd
import re
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
def ner_preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    ss_entity=[]
    se_entity=[]
    os_entity=[]
    oe_entity=[]
    sen_li=[]
    for i,j,sen in zip(dataset['subject_entity'], dataset['object_entity'],dataset['sentence']):
        sepi=i.split('start_idx')
        sepj=j.split('start_idx')
        ss,se=re.findall(r'[0-9]+',sepi[1])
        os,oe=re.findall(r'[0-9]+',sepj[1])
        i = i.split('\'type\': ')
        j = j.split('\'type\': ')
        i=re.search(r'[A-z]+',i[1]).group()
        j=re.search(r'[A-z]+',j[1]).group()
    #
        
        sen_li.append(sen)
        subject_entity.append(i)
        object_entity.append(j)
        ss_entity.append(ss)
        se_entity.append(se)
        os_entity.append(os)
        oe_entity.append(oe)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sen_li,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],'ss':ss_entity,'se':se_entity,'os':os_entity,'oe':oe_entity,})
    return out_dataset

def funct_preprocessing(dataset,mode="None"):
    new_tokens=[]
    idli=[]
    id=-1
    for sen,e01,e02, ss,se,os,oe in zip(dataset['sentence'], dataset['subject_entity'],dataset['object_entity'],dataset['ss'],dataset['se'],dataset['os'],dataset['oe']):
        id+=1
        temp = ''
        subj_start = '[SUBJ-{}]'.format(e01)
        subj_end='[/SUBJ-{}]'.format(e01)
        obj_start='[OBJ-{}]'.format(e02)
        obj_end='[/OBJ-{}]'.format(e02)
        ss=int(ss)
        se=int(se)
        os=int(os)
        oe=int(oe)
        for token in (subj_start,subj_end,obj_start,obj_end):
            if token not in new_tokens:    
                new_tokens.append(token)
                tokenizer.add_tokens([token])
    
        if ss<=os:
            temp=sen[0:ss]+subj_start+sen[ss:se+1]+subj_end+sen[se+1:os]+obj_start+sen[os:oe+1]+obj_end+sen[oe+1:] 
        else:
            temp=sen[0:os]+obj_start+sen[os:oe+1]+obj_end+sen[oe+1:ss]+subj_start+sen[ss:se+1]+subj_end+sen[se+1:]
        if mode=="pre":
            temp = re.sub(r"[^a-zA-Z0-9가-힣<>/.,!?\'\";:()%\-\@\*#^~]", " ", temp)
        elif mode=="None":
            temp=temp
        elif mode=="chinese":
            temp = re.sub(r'一-龥',"",temp)
            
        idli.append(id)
        concat_entity.append(temp)
    len_addtoken=len(new_tokens)
    print(new_tokens)
    out_dataset = pd.DataFrame({ 'sentence':concat_entity,'label':dataset['label'],},index=idli)  
    return out_dataset

def last_preprocessing(dataset,mode=None):
    concat_entity=[]
    len_addtoken=0

    match_dict={
    'PER': '사람',
    'ORG':'조직',
    'DAT':'날짜',
    'LOC':'장소',
    'POH':'단어',#직업이 많아보이긴함 근데 완전다른것도 존재
    'NOH':'숫자'
    }
    idli=[]
    subject_li=[]
    object_li=[]
    subject_idx_li=[]
    object_idx_li=[]
    subject_entity_li=[]
    object_entity_li=[]
    id=-1
    for sen,e01,e02, ss,se,os,oe in zip(dataset['sentence'], dataset['subject_entity'],dataset['object_entity'],dataset['ss'],dataset['se'],dataset['os'],dataset['oe']):
        id+=1

        temp = ''
        sub_entity=match_dict[e01]
        obj_entity=match_dict[e02]

        ss=int(ss)
        se=int(se)
        os=int(os)
        oe=int(oe)
        subject_entity=sen[ss:se+1]
        object_entity=sen[os:oe+1]    
        if ss<=os:
            temp=sen[0:ss]+"@*"+sub_entity+"*"+sen[ss:se+1]+"@"+sen[se+1:os]+"#^"+obj_entity+"^"+sen[os:oe+1]+"#"+sen[oe+1:] 
        else:
            temp=sen[0:os]+"#^"+obj_entity+"^"+sen[os:oe+1]+"#"+sen[oe+1:ss]+"@*"+sub_entity+"*"+sen[ss:se+1]+"@"+sen[se+1:]
        if mode=="pre":
            temp = re.sub(r"[^a-zA-Z0-9가-힣<>/.,!?\'\";:()%\-\@\*#^~]", " ", temp)
        elif mode=="None":
            temp=temp
        elif mode=="chinese":
            temp = re.sub(r'一-龥',"",temp)
        

        idli.append(id)
        concat_entity.append(temp)
        out_dataset = pd.DataFrame({ 'sentence':concat_entity,'label':dataset['label'],},index=idli)  
    return out_dataset

