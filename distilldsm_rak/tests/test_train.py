import unittest
import os
from src.utils.downloader import download_checkpoint, download_data
from src.utils.get_config import get_config
from src.utils.train_main import train_model 

def create_train_test():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train')
            cls.config = config
            if not os.path.exists(config["image_path"]):
                download_data()

        def test_trainer(self):
            if not os.path.exists(self.config["checkpoint"]):
                download_checkpoint()
            train_model(self.config)

        def test_config(self):
            self.config = get_config(action='train')

    return TrainerTest


class TestTrainer(create_train_test()):
    'Test case '

if __name__ == '__main__':

    unittest.main()