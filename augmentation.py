from koeda import AEDA
import pandas as pd
from tqdm import tqdm
import random

#apt-get update
#apt install -y sudo
#sudo apt install default-jdk

SPACE_TOKEN = "\u241F"

def replace_space(text):
    return text.replace(" ", SPACE_TOKEN)

def revert_space(text):
    clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
    return clean

#Improved AEDA library
class myAEDA(AEDA):
    def _aeda(self, data, p):
        if p is None:
            p = self.ratio

        split_words = self.morpheme_analyzer.morphs(replace_space(data))
        words = self.morpheme_analyzer.morphs(data)

        new_words = []
        q = random.randint(1, int(p * len(words) + 1))
        qs_list = [index for index in range(len(split_words)) if split_words[index] != SPACE_TOKEN]
        qs = random.sample(qs_list, q)

        for j, word in enumerate(split_words):
            if j in qs:
                new_words.append(SPACE_TOKEN)
                new_words.append(self.punctuations[random.randint(0, len(self.punctuations) - 1)])
                new_words.append(SPACE_TOKEN)
                new_words.append(word)
            else:
                new_words.append(word)

        augmented_sentences = revert_space(new_words)

        return augmented_sentences

def make_new_text(sentence, punc_ratio):
    aeda = myAEDA(morpheme_analyzer="Okt", punc_ratio=punc_ratio, punctuations=[".", ",", "!", "?", ";", ":"])
    new_sentence = aeda(sentence)

    return new_sentence

def append_new_sentence(new_df, train_df, i, sentence):
    new_df.loc[len(new_df)] = [
        train_df.loc[i]["id"],
        sentence,
        train_df.loc[i]["subject_entity"],
        train_df.loc[i]["object_entity"],
        train_df.loc[i]["subject_type"],
        train_df.loc[i]["object_type"],
        train_df.loc[i]["label"],
        train_df.loc[i]["subject_idx"],
        train_df.loc[i]["object_idx"],
    ]

def calculate_idx(dataset):
    new_sub_idx, new_obj_idx= [], []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):
        sub_start_idx = sen.find('@')
        sub_end_idx = sub_start_idx+(sub_idx[1]-sub_idx[0]+1)+6
        new_sub_i = [sub_start_idx, sub_end_idx]
        new_sub_idx.append(new_sub_i)
            
        obj_start_idx = sen.find('#')
        obj_end_idx = obj_start_idx+(obj_idx[1]-obj_idx[0]+1)+6
        new_obj_i = [obj_start_idx, obj_end_idx]
        new_obj_idx.append(new_obj_i)
    
    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : dataset['sentence'], 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': new_sub_idx, 'object_idx': new_obj_idx})
    return out_sentence

def random_delete(dataset, p):
    new_sentence= []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):
        if random.random() <= p:
            if len(sen) <= 2:
                new_sentence.append(sen)

            sub_start_idx = sen.find('@')
            sub_len = sub_idx[1]-sub_idx[0]+1
            tmp_sub = sen[sub_start_idx:sub_start_idx+sub_len]
            
            obj_start_idx = sen.find('#')
            obj_len = obj_idx[1]-obj_idx[0]+1
            tmp_obj = sen[obj_start_idx:obj_start_idx+obj_len]
            
            sen=sen.replace(tmp_sub,'@')
            sen=sen.replace(tmp_obj,'#')
            is_delete = False
            words = sen.split()
            while is_delete == False:
                delete_idx = random.randint(0,len(words)-1)
                if words[delete_idx] != '@' and words[delete_idx] != '#':
                    is_delete=True
                    del words[delete_idx]
                    sen=" ".join(words)
                    sen=sen.replace('@',tmp_sub)
                    sen=sen.replace('#',tmp_obj)
                    new_sentence.append(sen)

        else:
            new_sentence.append(sen)

    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : new_sentence, 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': dataset['subject_idx'], 'object_idx': dataset['object_idx']})
    return out_sentence

def aeda(train_df, train_label, check_num):
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
            new_label.append(train_label[i])

    aug_df = train_df.append(new_df, ignore_index=True)

    train_label.extend(new_label)

    return aug_df, train_label