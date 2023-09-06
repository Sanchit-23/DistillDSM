import unittest
import os
from downloader import download_checkpoint, download_data
from get_config import get_config
from train_main import train_model 

def create_train_test():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train')
            cls.config = config
            cls.paths = {
                'img_path': cls.config['image_path'],
                'mask_path': cls.config['mask_path'],
                'save_path': cls.config['save_path'],
                "data_split": cls.config['data_split']}
            if not os.path.exists(cls.paths['img_path']) or not os.path.exists(cls.paths['mask_path']):
                download_data()

        def test_trainer(self):
            if not os.path.exists(self.config["checkpoint"]):
                download_checkpoint()
            # self.device = self.config["device"]
            train_model(self.paths, self.config)            

        def test_config(self):
            self.config = get_config(action='train')


    return TrainerTest


class TestTrainer(create_train_test()):
    'Test case '

if __name__ == '__main__':

    unittest.main()