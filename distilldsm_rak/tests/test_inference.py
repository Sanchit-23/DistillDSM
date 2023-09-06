import os
import numpy as np
import torch
import unittest
from torch.utils.data import DataLoader
from src.utils.get_config import get_config
from src.utils.dataloader import PancreasDataLoader
from src.utils.downloader import download_data
from src.utils.inference_utils import inference_model
import json
import random
def create_inference_test():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference')

            if not os.path.exists(cls.config['image_path']):
                download_data()
            
            with open(cls.config['data_split']) as f3:
                data_split = json.load(f3)
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            
            tst_data_loader = PancreasDataLoader(cls.config['image_path'], cls.config['mask_path']
                                        ,data_split, split = 'valid')
            cls.tst_loader = DataLoader(tst_data_loader,
                                        batch_size=1, 
                                        shuffle=False,num_workers=8,
                                        pin_memory=True, worker_init_fn=seed_worker,
                                        drop_last=True)

        def test_pytorch_inference(self):

            config = get_config(action='inference')
            inference_model(config,'pytorch')
 

        def test_onnx_inference(self):
            config = get_config(action='inference')
            inference_model(config,'onnx')

           

        def test_ir_inference(self):
            config = get_config(action='inference')
            inference_model(config,'ir')

    return InferenceTest


class TestTrainer(create_inference_test()):
    'Test case for Inference'

if __name__ == '__main__':

    unittest.main()