from pororo import Pororo
import pandas as pd
from tqdm import tqdm
import re

mt = Pororo(task='translation', lang='multi')

def back_trans_pororo(original_text, lang='en'):
    text_to_lang = mt(original_text, src="ko", tgt=lang)
    new_text = mt(text_to_lang, src=lang, tgt="ko")
    return new_text

dataset_dir = '/opt/ml/github/NLP_dataset/train/train.csv'
df = pd.read_csv(dataset_dir)

new_ids, new_sentences, new_subject_entities, new_object_entities, new_labels, new_sources = [], [], [], [], [], []

print('Back Translating...')
for id, sentence, subject_entity, object_entity, lbl, so in tqdm(zip(df['id'], df['sentence'], df['subject_entity'], df['object_entity'], df['label'], df['source']), total=len(df)):
    new_subject, new_object = eval(subject_entity), eval(object_entity)
    regex = re.compile('[^ a-zA-Z0-9가-힇ㄱ-ㅎㅏ-ㅣ-!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]+')
    sentence_re = regex.sub('', sentence)
    new_sentence = back_trans_pororo(sentence_re)

    if new_sentence.find(new_subject['word']) >= 0 and new_sentence.find(new_object['word']) >= 0:
        new_subject['start_idx'] = new_sentence.index(new_subject['word'])
        new_subject['end_idx'] = new_subject['start_idx'] + len(new_subject['word']) - 1

        new_object['start_idx'] = new_sentence.index(new_object['word'])
        new_object['end_idx'] = new_object['start_idx'] + len(new_object['word']) - 1

        new_subject = str(new_subject)
        new_object = str(new_object)

        new_ids.append(id)
        new_sentences.append(new_sentence)
        new_subject_entities.append(new_subject)
        new_object_entities.append(new_object)
        new_labels.append(lbl)
        new_sources.append(so)
    
    

result = pd.DataFrame({'id':new_ids, 'sentence':new_sentences, 'subject_entity':new_subject_entities, 'object_entity':new_object_entities,
                       'label':new_labels, 'source':new_sources})

final = pd.concat([df, result])
print(df.shape, result.shape, final.shape)
final.to_csv('BT_train.csv')