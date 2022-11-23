from torch import nn
from transformers import Trainer,get_scheduler,TrainingArguments
from loss import *
import numpy as np
from typing import Union, Tuple, NamedTuple, Optional, Dict

def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)

    return tensors[:limit]


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()

def atleast_1d(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result

def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)

    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

# class RE_Trainer(Trainer):
#     def __init__(self, loss_name, 
#                        scheduler,
#                        num_training_steps, 
#                        *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.loss_name= loss_name
#         self.scheduler = scheduler
#         self.num_training_steps = num_training_steps
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)
#         if self.loss_name == 'CE':
#           loss_fct = nn.CrossEntropyLoss()
#         elif self.loss_name == 'LBS':
#           loss_fct = LabelSmoothingLoss()
#         elif self.loss_name == 'focal':
#           loss_fct = FocalLoss()
#         elif self.loss_name == 'f1':
#           loss_fct = F1Loss()
          
#         loss = loss_fct(logits.view(-1, 30), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss
    
#     def create_scheduler(self, num_training_steps, optimizer= None ):
#       if self.scheduler == 'linear' or self.scheduler == 'cosine':
#         if self.scheduler == 'linear':
#           my_scheduler = "linear"
#         elif self.scheduler == 'cosine':
#           my_scheduler = "cosine_with_restarts"

#         self.lr_scheduler = get_scheduler(
#             my_scheduler,
#             optimizer=self.optimizer if optimizer is None else optimizer,
#             num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
#             num_training_steps=num_training_steps,
#           )

#       elif self.scheduler == 'steplr':
#         self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1080, gamma=0.5)

#       return self.lr_scheduler



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
        outputs = model(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'],
                        attention_mask=inputs['attention_mask'], entity_ids=inputs['entity_ids'])
        logits = outputs['outputs']
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

