import torch
from torch.utils import data
import os
from torchvision import transforms
import numpy as np


class PancreasDataLoader(data.Dataset):

    def __init__(self, img_path, mask_path, data_split, split="train"):

        # self.split = split
        self.img_path = img_path
        self.mask_path = mask_path
        self.files = data_split[split]
        self.image_tf = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((128,128), antialias=None)])
        self.gt_tf = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((128,128), antialias=None)])

    def __len__(self):
        return len(self.files)
    
    def pad_channels(self,image):       
        C, H, W = image.size()
        
        if C < 145:
            num_channels_to_pad = int(145 - C)
            padded_channels = torch.zeros([ num_channels_to_pad, int(H),int(W)] , device=image.device)
            padded_image =torch.cat([image, padded_channels], dim=0)
        else:
            padded_image = image       
        return padded_image
    
    def __getitem__(self, index):
        filename = self.files[index]
        img = np.load(os.path.join(self.img_path,filename))
        mask = np.load(os.path.join(self.mask_path,filename))

        img, mask = self.transform(img, mask)
        img = self.pad_channels(img)
        mask = self.pad_channels(mask)

        return img.unsqueeze(0), mask.unsqueeze(0)
      

    def transform(self, img, mask):
        img = self.image_tf(img)
        img = img.type(torch.FloatTensor)
        mask = self.gt_tf(mask)
        mask = mask.type(torch.FloatTensor)

        return img, mask