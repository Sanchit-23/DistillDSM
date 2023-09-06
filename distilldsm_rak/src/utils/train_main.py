import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm as tq
import json
from src.utils.dataloader import PancreasDataLoader
from src.utils.distill_dsm import U_Net
from torchmetrics.classification import Dice
# For Reproducability
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.deterministic 
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth=1e-5):
        
        # y_pred = F.softmax(y_pred,dim=1)#[:, 1]
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        intersection = (y_pred * y_true).sum()
        dice = (2.*intersection + smooth)/(y_pred.sum() + y_true.sum() + smooth)
        return 1 - dice

#from torchmetrics.classification import BinaryF1Score
from monai.metrics import DiceMetric
# def dice_score(y_pred, y_true):
#     smooth = 1.
#     y_pred = y_pred.flatten()
#     y_true = y_true.flatten()
#     intersection = (y_pred * y_true).sum()
#     dice = (2.*intersection + smooth)/(y_pred.sum() + y_true.sum() + smooth)
#    return dice

def dice_score(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def max_channel(data_loader):
    max_tsm_length = float('-inf')
    for (img,mask) in data_loader:
        max_tsm_value_in_batch, _ = img.max(dim=3)  
        max_tsm_value_in_batch = max_tsm_value_in_batch.max()  

        if max_tsm_value_in_batch > max_tsm_length:
            max_tsm_length = max_tsm_value_in_batch
    return max_tsm_length.item()

def pad_channels(image):
    batch_size = int(1)   
    channel = int(1)       
    _, _, C, H, W = image.size()
    
    if C < 145:
        num_channels_to_pad = int(145 - C)
        padded_channels = torch.zeros([batch_size, channel, num_channels_to_pad, int(H),int(W)] , device=image.device)
        padded_image =torch.cat([image, padded_channels], dim=2)
    else:
        padded_image = image
    
    return padded_image  

def train_model(paths ,config):
    img_path = paths['img_path']
    mask_path = paths['mask_path']
    save_path = paths['save_path']
    data_split = paths['data_split']

    #dice_metric = DiceMetric(include_background=True, reduction="mean")
    # dice_score = Dice(average='micro').to('cuda')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(data_split) as f3:
        data_split = json.load(f3)

    train_d_set = PancreasDataLoader(img_path, mask_path,
                                        data_split, split="train")

    train_data_loader = data.DataLoader(train_d_set,
                                        batch_size=config['batch_size'], 
                                        shuffle=True,num_workers=8,
                                        pin_memory=True, worker_init_fn=seed_worker,
                                        drop_last=True)

    valid_d_set = PancreasDataLoader(img_path, mask_path,
                                        data_split, split="valid")

    valid_data_loader = data.DataLoader(valid_d_set,
                                        batch_size=1,shuffle=True,
                                        num_workers=8, pin_memory=True, 
                                        worker_init_fn=seed_worker,
                                        drop_last=True)    
    
    net_seg = U_Net(1, 1, conv_type='conv_2d', tsm=True, tsm_length=145, learn=True)
    # Max Value of Channel Slices for Train DataLoader and Valid DataLoader was found out to be 145
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        net_seg = net_seg.cuda()
        #dice_coefficient = dice_coefficient.cuda() 

    optimizer_seg = optim.Adam(net_seg.parameters(), lr=config['seg_lr'], weight_decay=config['w_decay'])
    criterion_seg = DiceLoss()
    epochs = config['epochs']

    bestValidDice = 0.0

    for epoch in range(epochs):
        
        print(f'Epoch: {str(epoch)}')
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        trainDice_r = 0
        validDice_r = 0

        net_seg.train(True)       
        
        for (img, mask) in tq(train_data_loader):
 
            if use_gpu:
                inputs = img.cuda()
                print(inputs.shape)
                    
            seg_out = net_seg(inputs)
            
            # net_out_sf = F.softmax(seg_out.data, dim=2)
            # print(net_out_sf.shape)
            # preds = torch.argmax(net_out_sf, dim=1)

            net_out_sigmoid = torch.sigmoid(seg_out.data)
            preds = (net_out_sigmoid > 0.5).int()
            del inputs
            mask = mask.cuda()
            print(mask.shape)
            mask = (mask > 0).int() 
            net_loss = criterion_seg(seg_out, mask)

            optimizer_seg.zero_grad()
            net_loss.backward()
            # print('Network Output Shape:', seg_out.shape)
            # print('Predictions Shape:', preds.shape)
            # print('Mask Shape:', mask.shape)
            # print('Loss:', net_loss.item())
            
            torch.cuda.empty_cache()

            optimizer_seg.step()
            
            trainRunningLoss += net_loss.item()
            
            train_dice = dice_score(preds.squeeze(dim=1), mask.squeeze(dim=1))
            train_valid_indices = ~torch.isnan(train_dice)
            if torch.any(train_valid_indices):
                train_dice = train_dice[train_valid_indices]
                train_dice = torch.mean(train_dice).item()
            else:
                train_dice = 0.0
            
            train_dice = torch.tensor(train_dice)
            train_dice = torch.nan_to_num(train_dice, nan=0.0)
            
            
            trainDice_r += torch.mean(train_dice).item()
            trainBatches += 1
            # if trainBatches > 1:
            #     break

        net_seg.eval()

        with torch.no_grad():
            for (img, mask) in tq(valid_data_loader):

                if use_gpu:
                    inputs = img.cuda()
                    mask = mask.cuda()
                    mask = (mask > 0).float() 
                inputs = pad_channels(inputs)
                mask = pad_channels(mask)
                
                seg_out = net_seg(inputs)
                net_out_sigmoid = torch.sigmoid(seg_out.data)
                preds = (net_out_sigmoid > 0.5).int()

                net_loss = criterion_seg(seg_out, mask)
                
                
                val_dice = dice_score(preds.squeeze(dim=1), mask.squeeze(dim=1))
                val_valid_indices = ~torch.isnan(val_dice)
                if torch.any(val_valid_indices):
                    val_dice = val_dice[val_valid_indices]
                    val_dice = torch.mean(val_dice).item()
                else:
                    val_dice = 0.0
                val_dice = torch.tensor(val_dice)
                val_dice = torch.nan_to_num(val_dice, nan=0.0)
                validDice_r += torch.mean(val_dice).item()
                validRunningLoss += net_loss.item()
                validBatches += 1

        net_train_loss= trainRunningLoss/trainBatches
        net_valid_loss= validRunningLoss/validBatches
        train_dice= trainDice_r/trainBatches
        valid_dice= validDice_r/validBatches

        if valid_dice > bestValidDice:
            bestValidDice = valid_dice
            torch.save(net_seg.state_dict(), save_path+'model_best_seg.pt')

        print(f'Epoch: {epoch} | Train Dice: {train_dice} | Valid dice: {valid_dice}')

    torch.save(net_seg.state_dict(), save_path+'model_last_seg.pt')

    return net_train_loss,net_valid_loss,train_dice,valid_dice

if __name__=="__main__":

    paths ={
        'img_path': '/home/sanchit/bmi7_new_implentation/dataset/New_Dataset/img_volumes',
        'mask_path': '/home/sanchit/bmi7_new_implentation/dataset/New_Dataset/labels',
        'data_split': '/home/sanchit/bmi7_new_implentation/distilldsm_rak/configs/data_split.json',
        'save_path': '/home/sanchit/bmi7_new_implentation/distilldsm_rak/model_weights',
        
    }

    config = {
        'seg_lr': 1e-7,
        'batch_size': 4,
        'epochs':10,
        'w_decay': 1e-5

    }

    train_model(paths,config)