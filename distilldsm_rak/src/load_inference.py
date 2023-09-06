import torch
import onnxruntime
import os
from openvino.inference_engine import IECore
from utils.distill_dsm import U_Net


def load_inference_model(config , run_type):

    if run_type == 'pytorch':
        model = U_Net(1, 1, conv_type='conv_2d', tsm=True, tsm_length=145, learn=True)

        with open(os.path.abspath(config['model_file']), 'rb') as modelfile:
            loaded_model_file = torch.load(modelfile, map_location=torch.device('cpu'))
            model.load_state_dict(loaded_model_file['model_state'])
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