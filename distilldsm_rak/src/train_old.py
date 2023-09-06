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

# For Reproducability
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

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth=1e-5):
        
        y_pred = F.softmax(y_pred,dim=1)[:, 1]
        y_pred = y_pred.flatten()
        y_true = y_true[:,1].flatten()
        intersection = (y_pred * y_true).sum()
        dice = (2.*intersection + smooth)/(y_pred.sum() + y_true.sum() + smooth)
        return 1 - dice

def dice_coefficient(pred1, target):
    smooth = 1e-15
    pred = torch.argmax(pred1, dim=1)
    num = pred.size()[0]
    pred_1_hot = torch.eye(3)[pred.squeeze(1)].cuda()
    pred_1_hot = pred_1_hot.permute(0, 3, 1, 2).float()

    target_1_hot = torch.eye(3)[target].cuda()
    target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()

    m1_1 = pred_1_hot[:, 1, :, :].view(num, -1).float()
    m2_1 = target_1_hot[:, 1, :, :].view(num, -1).float()
    m1_2 = pred_1_hot[:, 2, :, :].view(num, -1).float()
    m2_2 = target_1_hot[:, 2, :, :].view(num, -1).float()

    intersection_1 = (m1_1*m2_1).sum(1)
    intersection_2 = (m1_2*m2_2).sum(1)
    union_1 = (m1_1+m2_1).sum(1) + smooth - intersection_1
    union_2 = (m1_2+m2_2).sum(1) + smooth - intersection_2
    score_1 = intersection_1/union_1

    return [score_1.mean()]

def train_model(paths,config):
    img_path = paths['img_path']
    mask_path = paths['mask_path']
    save_path = paths['save_path']
    data_split = paths['data_split']

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(data_split) as f3:
        data_split = json.load(f3)

    net_seg = U_Net(1, 1, conv_type='conv_2d', tsm=True, alpha=1, tsm_length=78, learn=True)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        net_seg = net_seg.cuda()

    optimizer_seg = optim.Adam(net_seg.parameters(),
                               lr=config['seg_lr'],
                               weight_decay= config['w_decay'])
    
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

        for (img, mask) in tq(train_data_loader):

            if use_gpu:
                inputs = img.cuda()
                mask = mask.cuda()
            print(inputs.shape)
            print(mask.shape)

            seg_out = net_seg(inputs)
            net_out_sf = F.softmax(seg_out.data, dim=1)
            preds = torch.argmax(net_out_sf, dim=1)
            net_loss = criterion_seg(seg_out, mask)

            optimizer_seg.zero_grad()
            net_loss.backward()

            optimizer_seg.step()
            trainRunningLoss += net_loss.item()
            train_dice = dice_coefficient(preds, mask[:,1])
            trainDice_r += torch.mean(train_dice).item()

            trainBatches += 1
            # if trainBatches > 1:
            #     break

        net_seg.eval()

        with torch.no_grad():
            for (img, mask) in tq(valid_data_loader):

                if use_gpu:
                    img = img.cuda()
                    mask = mask.cuda()

                seg_out = net_seg(img)
                net_out_sf = F.softmax(seg_out.data, dim=1)
                preds = torch.argmax(net_out_sf, dim=1)

                net_loss = criterion_seg(seg_out, mask)

                val_dice = dice_coefficient(preds, mask[:,1])
                validDice_r += torch.mean(val_dice).item()
                validRunningLoss += net_loss.item()
                validBatches += 1
                # if validBatches > 1:
                #     break

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
        'img_path': '/home/deeptensor/rakshith_codes/bmi7_new_implentation/dataset/img_volumes/',
        'mask_path': '/home/deeptensor/rakshith_codes/bmi7_new_implentation/dataset/labels/',
        'data_split': '/home/deeptensor/rakshith_codes/bmi7_new_implentation/data_split.json',
        'save_path': '/home/deeptensor/rakshith_codes/bmi7_new_implentation/distilldsm_rak/model_weights/'
    }

    config = {
        'seg_lr': 1e-4,
        'batch_size': 1,
        'epochs':30,
        'w_decay': 1e-5

    }

    train_model(paths,config)