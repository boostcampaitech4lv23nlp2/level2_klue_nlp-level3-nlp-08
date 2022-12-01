import argparse
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
def make_new_text(sentence, punc_ratio, subject_entity, object_entity):
    aeda = myAEDA(morpheme_analyzer="Okt", punc_ratio=punc_ratio, punctuations=[".", ",", "!", "?", ";", ":"])
    #Entity 사이에 punctutation이 안들어가는 경우의 수만 return
    while True:
        new_sentence = aeda(sentence)
        if subject_entity in new_sentence and object_entity in new_sentence:
            break
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
    parser.add_argument('--train_file', type=str, default='BT_AEDA_train_final.csv')
    args = parser.parse_args()

    train_path = '../NLP_dataset/train/BT_train_preprocess_final.csv'
    train_dataset = pd.read_csv(train_path, index_col=0)
    aeda(args, train_dataset, 2)

