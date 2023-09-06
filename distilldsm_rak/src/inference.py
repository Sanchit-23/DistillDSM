import torch
from torch.utils import data
import torch.nn.functional as F
import os
from tqdm import tqdm as tq
import json
from utils.metrics import dice_coefficient
from utils.dataloader import PancreasDataLoader
from distill_dsm import DistillDSM

def load_checkpoint(model, checkpoint):
    if checkpoint is not None:
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint)
    else:
        model.state_dict()
    return model

def inference(paths,config):
    data_path = paths['datapath']
    save_path = paths['savepath']
    data_split = paths['data_split']

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(data_split) as f3:
        data_split = json.load(f3)

    net_seg = DistillDSM()
    use_gpu = torch.cuda.is_available()
    net = load_checkpoint(net, save_path+'_best.pt')
    

    if use_gpu:
        net_seg = net_seg.cuda()

    test_d_set = PancreasDataLoader(datapath=data_path,
                                         is_transform=True,
                                         seg_json=data_split)

    test_data_loader = data.DataLoader(test_d_set,
                                        batch_size=1,shuffle=True,
                                        num_workers=8, pin_memory=True,
                                        drop_last=True)

    for (img, mask) in tq(test_data_loader):

        if use_gpu:
            img = img.cuda()
            mask = mask.cuda()

        seg_out = net_seg(img)
        net_out_sf = F.softmax(seg_out.data, dim=1)
        preds = torch.argmax(net_out_sf, dim=1)

        test_dice = dice_coefficient(preds, mask[:,1])
        testDice_r += torch.mean(test_dice).item()
        testBatches += 1
        # if testBatches > 1:
        #     break

    test_dice = testDice_r/testBatches

    return test_dice