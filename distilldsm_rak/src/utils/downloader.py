from src.utils.get_config import get_config
import os
import zipfile
import wget

def download_and_extract(path, url, expath):
    wget.download(url, path)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_checkpoint():
    config = get_config(action='download', config_path='configs/')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    url = config['model']['url_model']
    path = config['model']['dest_path_model']
    download_and_extract(path=path, url=url,
                         expath='model_weights/')
    
def download_data():
    config = get_config(action='download', config_path='configs/')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    data_url = config['test_data']['url_data']
    data_path = config['test_data']['dest_path_data']
    download_and_extract(path=data_path, url=data_url, expath='test_data/')


    
