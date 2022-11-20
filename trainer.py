from torch import nn
from transformers import Trainer,get_scheduler,TrainingArguments
from loss import *


class RE_Trainer(Trainer):
    def __init__(self, loss_name, 
                       scheduler,
                       num_training_steps, 
                       *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name
        self.scheduler = scheduler
        self.num_training_steps = num_training_steps
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        if self.loss_name == 'CE':
          loss_fct = nn.CrossEntropyLoss()
        elif self.loss_name == 'LBS':
          loss_fct = LabelSmoothingLoss()
        elif self.loss_name == 'focal':
          loss_fct = FocalLoss()
        elif self.loss_name == 'f1':
          loss_fct = F1Loss()
          
        loss = loss_fct(logits.view(-1, 30), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def create_scheduler(self, num_training_steps, optimizer= None ):
      if self.scheduler == 'linear' or self.scheduler == 'cosine':
        if self.scheduler == 'linear':
          my_scheduler = "linear"
        elif self.scheduler == 'cosine':
          my_scheduler = "cosine_with_restarts"

        self.lr_scheduler = get_scheduler(
            my_scheduler,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
          )

      elif self.scheduler == 'steplr':
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1080, gamma=0.5)

      return self.lr_scheduler
    