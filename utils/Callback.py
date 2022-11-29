from transformers import TrainerCallback
from sklearn.manifold import TSNE
import wandb
import plotly.express as px 
import wandb
import pandas as pd
from inference import num_to_label

class CustomCallback(TrainerCallback):
    def __init__(self):
        self.eval_step = 0
        self.tSNE = TSNE(n_components=2, perplexity = 40)
        self.html_list = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        pass
        
    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_step += 1
        embedding = state.log_history[-1]['eval_embedding']
        labels = state.log_history[-1]['eval_labels']
        answer = state.log_history[-1]['eval_answer'] 
        preds = state.log_history[-1]['eval_preds']  
        # 안하면 이후 직렬화 오류남.
        del(state.log_history[-1]['eval_embedding'])
        del(state.log_history[-1]['eval_labels'])
        del(state.log_history[-1]['eval_answer'])
        del(state.log_history[-1]['eval_preds'])
        
        str_labels = num_to_label(range(30))
        data = [[label, val] for (label, val) in zip(str_labels, answer)]
        acc_table = wandb.Table(data=data, columns = ["label", "acc"])
        wandb.log({"acc_table": acc_table})

        tSNE_result = self.tSNE.fit_transform(embedding)
        df = pd.DataFrame()
        df['x'] = tSNE_result[:, 0]
        df['y'] = tSNE_result[:, 1]
        df['label'] = num_to_label(labels)
        df['pred'] = num_to_label(preds)
        fig =px.scatter(df,
                        x='x',
                        y='y',
                        color='label',
                        hover_data=['pred'],
                        size=None            
                )
        # TODO __init__에서 cfg 불러와서 경로 로드
        path_to_plotly_html = f"EDA/tSNE.html"
        fig.write_html(path_to_plotly_html, auto_play = False)
        self.html_list.append(wandb.Html(path_to_plotly_html))
        # self.html_list이 커질 경우 느려지는 문제가 있음.
        wandb.log({"t-SNE": wandb.Table(data = [[i] for i in self.html_list], 
                                        columns = ['tSNE'])})
