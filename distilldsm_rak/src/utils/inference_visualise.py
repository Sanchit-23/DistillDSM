import torch
import torchvision
import os
from torch import nn
import onnxruntime
from src.utils.distill_dsm import U_Net
from src.utils.dataloader import PancreasDataLoader
from torch.utils import data
import numpy as np
import pickle
import sys
from PIL import Image
from openvino.inference_engine import IECore
from torch.backends import cudnn
import json
import random
from tqdm import tqdm as tq
from torchvision import transforms
from src.utils.train_main import DiceLoss
from src.utils.train_main import dice_score
from src.utils.get_config import get_config
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
cudnn.benchmark = True

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def load_inference_model(config , run_type):

    if run_type == 'pytorch':
        model = U_Net(1, 1, conv_type='conv_2d', tsm=True, tsm_length=145, learn=True)
        loaded_model_file = torch.load(config['model_file'])
        model.load_state_dict(loaded_model_file)
        
        model.eval()

    elif run_type == 'onnx':
        model = onnxruntime.InferenceSession(os.path.splitext(config['checkpoint'])[0] + ".onnx")

    else:
        ie = IECore()
        split_text = os.path.splitext(config['checkpoint'])[0]
        model_xml =  split_text + ".xml"
        model_bin = split_text + ".bin"
        model_temp = ie.read_network(model_xml, model_bin)
        model = ie.load_network(network=model_temp, device_name='CPU')

    return model

def visualize_sample(input_image, mask, prd_final):
    # Loop over each sample in the batch
    for i in range(input_image.size(0)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Convert PyTorch tensors to NumPy arrays and resize to (128, 128)
        input_image_i = TF.to_pil_image(input_image[i].squeeze(0))
        input_image_i = TF.resize(input_image_i, (128, 128))
        input_image_i = TF.to_tensor(input_image_i)

        mask_i = TF.to_pil_image(mask[i].squeeze(0))
        mask_i = TF.resize(mask_i, (128, 128))
        mask_i = TF.to_tensor(mask_i)

        prd_final_i = prd_final[i]  # Remove the unnecessary squeeze(0) call

        # Print the shape of prd_final_i before visualization
        print("Shape of prd_final_i:", prd_final_i.shape)


        # Visualize the images
        axes[0].imshow(TF.to_pil_image(input_image_i), cmap='gray')
        axes[0].set_title("Input Image")

        axes[1].imshow(TF.to_pil_image(mask_i), cmap='gray')
        axes[1].set_title("Ground Truth Mask")

        axes[2].imshow(TF.to_pil_image(prd_final_i), cmap='gray')
        axes[2].set_title("Model Output")

        plt.show()

def validate_model(model, config, run_type):

    if run_type == 'pytorch':
        if torch.cuda.is_available() and config['gpu'] == 'True':
            model = model.cuda()
           
    else:
        pass
    with open(config['data_split']) as f3:
        data_split = json.load(f3)

    infer_d_set = PancreasDataLoader(config['image_path'], config['mask_path'],
                                        data_split, split='valid')

    infer_data_loader = data.DataLoader(infer_d_set,
                                        batch_size=config['batch_size'], 
                                        shuffle=True,num_workers=8,
                                        pin_memory=True, worker_init_fn=seed_worker,
                                        drop_last=True)
    
    criterion = DiceLoss()
    validDice_r = 0
    validRunningLoss = 0
    validBatches = 0

    target_size = (128, 128)
    with torch.no_grad():
        n = 0
        for (img, mask) in tq(infer_data_loader):
            if config['gpu'] == 'True':
                    img = img.cuda()
                    mask = mask.cuda()
                    gt = (mask > 0).float()

            if run_type == 'pytorch':
                prd_final = model(img)
                prd_final = prd_final.squeeze(0)

            
            elif run_type == 'onnx':
                ort_inputs = {model.get_inputs()[0].name: to_numpy(img)}
                prd_final = model.run(None, ort_inputs)
                to_tensor = transforms.ToTensor()
                prd_final = np.array(prd_final)
                prd_final = np.squeeze(prd_final,axis=0)
                prd_final = to_tensor(prd_final).unsqueeze(0)
                prd_final = prd_final.squeeze(1).transpose(1,0)
                gt = gt.cpu()
            else:
                to_tensor = transforms.ToTensor()
                prd_final = model.infer(inputs={'input': img.cpu()})['output']
                prd_final = np.array(prd_final)
                prd_final = np.squeeze(prd_final,axis=0)
                prd_final = to_tensor(prd_final).unsqueeze(0)
                prd_final = prd_final.squeeze(0)
                gt=gt.cpu()
            
            net_out_sigmoid = torch.sigmoid(prd_final.data)
            prd_final = (net_out_sigmoid > 0.5).int()
            
            loss = criterion(prd_final,gt)

            val_dice = dice_score(prd_final, gt)
            validDice_r += torch.mean(val_dice).item()
            validRunningLoss += loss.item()
            validBatches += 1
            
            n +=1
            
            if n == config['max_samples']:
                break
           
            visualize_sample(img[0], mask[0], prd_final[0])
        net_valid_loss= validRunningLoss/validBatches
        valid_dice= validDice_r/validBatches
        
        
    
    return net_valid_loss , valid_dice

def inference_model(config, run_type):
    model = load_inference_model(config, run_type)
    val_loss , val_dice= validate_model(model, config, run_type)
    print (val_loss , val_dice)

if __name__ == '__main__':
    

    action = 'inference'  # Specify the desired action: 'download' or 'inference'
    run_type = 'pytorch'

    config = get_config(action,config_path='distilldsm_rak/configs/model_configs.json')
    inference_model(config, run_type)