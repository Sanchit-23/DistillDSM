import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm as tq
import json
from utils.dataloader import PancreasDataLoader
from utils.distill_dsm import U_Net
import random

random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

img_path = '/home/sanchit/bmi7_new_implentation/dataset/pancreas_data_volumes/img_volumes'
mask_path ='/home/sanchit/bmi7_new_implentation/dataset/pancreas_data_volumes/labels'
data_split: '/home/sanchit/bmi7_new_implentation/data_split.json'

train_d_set = PancreasDataLoader(img_path, mask_path,
                                         data_split)

train_data_loader = data.DataLoader(train_d_set,
                                     
                                    shuffle=True,num_workers=8,
                                    pin_memory=True, worker_init_fn=seed_worker,
                                    drop_last=True)   
for (img, mask) in tq(train_data_loader):
    print(img.shape)
    print(mask.shape)