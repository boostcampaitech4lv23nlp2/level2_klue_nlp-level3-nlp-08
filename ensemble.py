import os
import argparse

from tqdm import tqdm
from collections import Counter
from utils.metric import *
from sklearn.metrics import accuracy_score

from itertools import product
import pandas as pd

from EDA import NLPAnalyzer
from omegaconf import OmegaConf


def show_result(preds, probs, labels):
    print(f'f1: {klue_re_micro_f1(preds, labels):.3f}')   
    print(f'acc: {100*accuracy_score(preds, labels):.3f}') 
    print(f'auprc: {klue_re_auprc(probs, labels):.3f}\n') 
        


class EnsembleTool():
    def __init__(self, cfg):
        self.cfg = cfg
        self.analyzer = NLPAnalyzer(tokenizer=None, dict_label_to_num_path=self.cfg.dict_label_to_num_path)

        # 2. 라벨 타입 설정. (auxilary feature 전처리용도)
        self.analyzer.annotate_feature(col='label', type='label')
        self.analyzer.annotate_feature(col='output_prob', type='pred')
        self.analyzer.annotate_feature(col='sentence', type='sentence')

        # 3. 분석할 csv가 들어간 데이터 폴더 설정. put으로 파일 한 개만 추가가능.
        self.analyzer.puts(self.cfg.path)

        with open(self.cfg.dict_label_to_num_path, 'rb') as f:
            self.label_num = len(pickle.load(f))

        # hyper_param 
        self.allow_recall = self.cfg.allow_recall
        self.allow_precision = self.cfg.allow_precision 
        self.allow_all = self.cfg.allow_all
        self.w_acc = self.cfg.w_acc    
        self.w_auc = self.cfg.w_auc    
        self.w_f1 =  self.cfg.w_f1   
        self.weight_ticks = self.cfg.tick 
        self.weight_soft_bias = self.cfg.bias  
        
        self.label_list = self.analyzer.num_to_label(range(self.label_num))
        self.df_list = list(self.analyzer.gets().values())
        self.csv_num = len(self.df_list)

    def micro_klue_re_auprc(self, probs, labels):
        count = Counter(labels)
        valid = np.unique(labels)
        labels = np.eye(self.label_num)[labels]
        
        result = 0
        for c in valid:
            targets_c = labels.take([c], axis=1).ravel()
            preds_c = probs.take([c], axis=1).ravel()
            
            precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
            auc = sklearn.metrics.auc(recall, precision)
            result += auc*count[c]/len(labels)

        return result*100

    def show_result_df(self, df):
        labels = np.array(self.analyzer.label_to_num(df['label']))
        probs = np.array(list(map(lambda x: x.split(), df['output_prob']))).astype(np.float64)
        preds = np.array(np.argmax(probs, axis=-1))
        show_result(preds, probs, labels)
    
    def weights_optimize(self):
        # weight 세팅
        weights_param = product(range(self.weight_soft_bias, self.weight_soft_bias + self.weight_ticks), 
                                repeat=self.csv_num)
        weights_total = self.weight_ticks + self.weight_soft_bias*self.csv_num
        weights_param = [w for w in weights_param if sum(w) == weights_total]
        weights_param.insert(0, [1]*self.csv_num) # soft voting

        # probs 초기화 (csv수, row, num of class)
        best_weights = []
        df_list_probs = np.empty((self.csv_num, self.df_list[0].shape[0], self.label_num))
        for idx, df in enumerate(self.df_list):
            df_list_probs[idx] = np.array(list(map(lambda x: x.split(), df['output_prob']))).astype(np.float64)
            

        for num_label, check_label in tqdm(enumerate(self.label_list)):
            # TP, FN 측정용
            label_filter = np.where(self.df_list[0]['label'] == check_label)[0]
            for i, df in enumerate(self.df_list):
                # FP 측정용
                if i == 0:
                    pred_filter = np.where(df['pred'] == check_label)[0]
                else:
                    pred_filter = np.concatenate([pred_filter, np.where(df['pred'] == check_label)[0]])
            # 과반수 얻은 pred만 사용
            pred_filter = [i for i, c in Counter(pred_filter).items() if c >=  self.csv_num/2]
            # pred + label
            if self.allow_all:
                filter = np.array(range(self.csv_num))
            elif self.allow_recall and self.allow_precision: 
                filter = np.concatenate([label_filter, pred_filter])
                filter = np.unique(filter)  
            elif self.allow_recall:
                filter = label_filter
            else:
                filter = pred_filter
            print(f'num of samples : {len(filter)}')    
            labels = np.array(self.analyzer.label_to_num(self.df_list[0]['label']))[filter]
            best_score = 0
            best_weight = None
            for weight in weights_param:
                # softvoting으로 기준 잡음.
                probs = np.average(df_list_probs[:, filter], weights=weight, axis=0)  
                pred = np.argmax(probs, axis=-1)
                acc = accuracy_score(pred, labels) * 100 if self.w_acc else 0
                auc = self.micro_klue_re_auprc(probs, labels) if self.w_auc else 0
                f1 = klue_re_micro_f1(pred, labels) if self.w_f1 and num_label !=0 else 0  
                score = (acc*self.w_acc + auc*self.w_auc + f1*self.w_f1) / (self.w_acc + self.w_auc + self.w_f1)
                if best_score < score:
                    print(f'({check_label}), best weights are updated to {weight} / '
                        + f'{best_score:.2f} -> {score:.2f} [acc {acc:.3f}, auc {auc:.3f}, f1 {f1:.3f}]')
                    best_score = score
                    best_weight = weight
                    
            if self.allow_all and num_label == 1:
                best_weights.extend([best_weight]*(self.label_num - 2))
                break
            else:
                best_weights.append(best_weight)
        
        print('best weights / model_name :', list(self.analyzer.gets().keys()))    
        for i, label in enumerate(self.label_list):
            print(label, best_weights[i])
        
        return best_weights
        
    def get_results(self, best_weights):
        print('\nResult\n')
        # origin result
        if self.cfg.origin:
            for name, df in self.analyzer.gets().items():
                print(name)
                self.show_result_df(df)

        # probs 초기화 (csv수, row, num of class)
        df_list_probs = np.empty((self.csv_num, self.df_list[0].shape[0], self.label_num))
        for idx, df in enumerate(self.df_list):
            df_list_probs[idx] = np.array(list(map(lambda x: x.split(), df['output_prob']))).astype(np.float64)
                
        # hard voting
        if self.cfg.hard_voting:
            print('hard_voting')
            final_df = self.df_list[0].copy()
            for idx in tqdm(range(len(final_df))):
                voting_label = Counter([df['pred'].iloc[idx] for df in self.df_list]).most_common()[0][0]
                max_confidence = 0
                max_df = None
                for df in self.df_list:
                    if df['pred'].iloc[idx] != voting_label:
                        continue
                    pred_confidence = df['pred.confidence'].iloc[idx]
                    if pred_confidence > max_confidence:
                        max_confidence = pred_confidence
                        max_df = df
                final_df.loc[idx] = max_df.loc[idx]
            self.show_result_df(final_df)

        if self.cfg.soft_voting:
            print('soft_voting')
            final_df = self.df_list[0].copy()
            
            # soft probs (row, num of class)
            soft_probs = np.average(df_list_probs, axis=0)
            final_df['output_prob'] = [' '.join(map(lambda x: str(x), soft_prob)) for soft_prob in soft_probs]
            final_df['pred'] = np.array(np.argmax(soft_probs, axis=-1))
            self.show_result_df(final_df)


        if self.cfg.weighted_voting:
            print('weighted voting')
            final_df = self.df_list[0].copy()
            # soft probs (row, num of class)
            soft_probs = np.average(df_list_probs, axis=0)
            
            voting_label = np.argmax(soft_probs, axis=-1)
            weighted_probs = []
            for idx in tqdm(range(len(final_df))):
                label = voting_label[idx]
                weighted_soft_prob = np.average(df_list_probs[:, idx, :], weights=best_weights[label], axis=0)
                weighted_probs.append(weighted_soft_prob)
            final_df['output_prob'] = [' '.join(map(lambda x: str(x), prob)) for prob in weighted_probs]
            final_df['pred'] = np.array(np.argmax(weighted_probs, axis=-1))
            self.show_result_df(final_df) 
            # 앙상블 모델을 또 앙상블할 경우 사용. but 성능 잘 안나오는 듯.
            #final_df.to_csv('EDA/output/ensemble/ensemble.csv')
    
    def test_submission(self, best_weights):
        test_df_list = []
        for name in self.analyzer.gets().keys(): 
            test_df_list.append(pd.read_csv(f'{self.cfg.test_path}/{name}.csv'))
            
        df_list_probs = np.empty((self.csv_num, test_df_list[0].shape[0], self.label_num))
        for idx, df in enumerate(test_df_list):
            df_list_probs[idx] = np.array(list(map(lambda x: x.strip('[]').split(','), df['probs']))).astype(np.float64)
            
        if self.cfg.soft_voting:
            soft_df = test_df_list[0].copy()
            soft_probs = np.average(df_list_probs, axis=0)
            soft_df['probs'] = ['[' + ', '.join(map(lambda x: str(x), soft_prob)) +']' for soft_prob in soft_probs]
            soft_df['pred_label'] = self.analyzer.num_to_label(np.argmax(soft_probs, axis=-1))
            soft_df.to_csv('predict/soft-voting_submission.csv', index = False)
            
        if self.cfg.weighted_voting:
            weighted_df = test_df_list[0].copy()
            soft_probs = np.average(df_list_probs, axis=0)
            voting_label = np.argmax(soft_probs, axis=-1)
            weighted_probs = []
            for idx in tqdm(range(len(weighted_df))):
                label = voting_label[idx]
                weighted_soft_prob = np.average(df_list_probs[:, idx, :], weights=best_weights[label], axis=0)
                weighted_probs.append(weighted_soft_prob)
            weighted_df['probs'] = ['[' + ', '.join(map(lambda x: str(x), prob)) +']' for prob in weighted_probs]
            weighted_df['pred_label'] = self.analyzer.num_to_label(np.argmax(weighted_probs, axis=-1))
            weighted_df.to_csv('predict/weighted-voting_submission.csv', index = False)
            
            
        if self.cfg.soft_voting and self.cfg.weighted_voting:
            print('다른 라벨들 : ')
            filter = (weighted_df['pred_label'] != soft_df['pred_label'])
            show_df = pd.DataFrame()
            show_df['soft'] =  soft_df['pred_label'][filter]
            show_df['weighted'] =  weighted_df['pred_label'][filter]
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(show_df)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    # ensemble config가 있을 경우 사용
    parser.add_argument('--config', type=str, default=None)

    parser.add_argument('--path', type=str, default='EDA/output/ensemble') # 모든 모델에 동일하게 배분할 weight tick. 클수록 soft voting 효과가 커짐.
    parser.add_argument('--test_path', type=str, default='EDA/output/ensemble_test') # 모든 모델에 동일하게 배분할 weight tick. 클수록 soft voting 효과가 커짐.
  
    parser.add_argument('--dict_label_to_num_path', type=str, default='dict_label_to_num.pkl')

    parser.add_argument('--allow_recall', type=bool, default=True)  # 라벨 별 weight 계산 시 정답 샘플을 사용. TP,FN 체크 가능.
    parser.add_argument('--allow_precision', type=bool, default=True) # 라벨 별 weight 계산 시 모델들이 기준 라벨로 과반수 예측한 샘플을 사용.
    parser.add_argument('--allow_all', type=bool, default=False) # 라벨 별 weight 계산 시, 모든 샘플을 사용. 모든 기준 라벨에 대해 같은 weight값이 나옴.
    parser.add_argument('--w_acc', type=int, default=0) # 스코어 계산 시 acc weight. 
    parser.add_argument('--w_auc', type=int, default=1) # 스코어 계산 시 AUPRC weight. 높이면 auc이 보통 올라감. 0이면 계산속도 빨라짐.
    parser.add_argument('--w_f1', type=int, default=2) # 스코어 계산 시 f1 weight (no_relation에서는 0으로 강제 설정). 높이면 f1이 보통 올라감. 
    parser.add_argument('--tick', type=int, default=5) # 분배할 weight tick 수, 클수록 정교해지나 시간 오래걸림.
    parser.add_argument('--bias', type=int, default=2) # 모든 모델에 동일하게 배분할 weight tick. 클수록 soft voting 효과가 커짐.

    # 결과 선택적 출력
    parser.add_argument('--origin', type=bool, default=True)
    parser.add_argument('--hard_voting', type=bool, default=False)
    parser.add_argument('--soft_voting', type=bool, default=True)
    parser.add_argument('--weighted_voting', type=bool, default=True)


    cfg , _ = parser.parse_known_args()
    if cfg.config != None:
        cfg = OmegaConf.load(cfg.config)
        
    ensembleTool = EnsembleTool(cfg)
    
    if cfg.weighted_voting:
        best_weights = ensembleTool.weights_optimize()
    else:
        best_weights = None
    ensembleTool.get_results(best_weights)
    ensembleTool.test_submission(best_weights)
        






