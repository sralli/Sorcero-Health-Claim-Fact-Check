# This file has all the classes/functions needed for the model to be loaded. 

#Loading libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
import numpy as np
import random, os
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging,
)

from sklearn.utils.class_weight import compute_class_weight

## Dataloader helper function to prepare data
def prepare_train_features(args, row, tokenizer):
    tokenized_example = tokenizer(
        row["claim"],
        row["top_k_sentences_joined"],
        truncation='longest_first',
        max_length=args.max_seq_length,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )


    features = []
  
    feature = {}

    input_ids = tokenized_example["input_ids"].squeeze()
    attention_mask = tokenized_example["attention_mask"].squeeze()

    feature['input_ids'] = input_ids
    feature['attention_mask'] = attention_mask
    feature['label'] = row['label']
    features.append(feature)
    return features



# Create Scheduler and Optimizer 

def make_scheduler(
    args, optimizer, 
    num_warmup_steps, 
    num_training_steps
):
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler    


def make_optimizer(args, model):
    '''
        The function to create an optimizer for the training process. 

        - Args: 
            - @param: args: dict, the training parameters defined in model config
            - @param: model: The loaded model

        - Returns: 
            - optimizer: Returns the optimizer needed for the training. 
    
    '''
   
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer_type == "AdamW":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,

        )
        return optimizer


# Eval metrics class: weights 

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


# Set all seeds 

def fix_all_seeds(seed):
    '''
        Fixes the seed for all variables in the code

        - Args:
            @param: seed : int. The seed value to be ser

    
    '''
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_class_weight(labels):
    '''
        This function returns the class weight for labels

        - Args: labels: list/tensor with the classes

        - Returns: the class weight in a tensor
    
    '''


    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    
    return class_weights.cuda()
