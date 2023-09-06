import torch
import os
from src.utils.distill_dsm import U_Net
from src.utils.distill_dsm import load_checkpoint 
class Exporter:
    def __init__(self, config):
        self.config = config
        self.checkpoint = config.get('checkpoint')
        self.model = U_Net(1, 1, conv_type='conv_2d', tsm=True, tsm_length=145, learn=True)
        self.model.eval()
        load_checkpoint(self.model, self.checkpoint)

    def export_model_ir(self):
        input_model = os.path.join(
            os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))
        input_shape = self.config.get('input_shape')
        output_dir = os.path.split(self.checkpoint)[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir}"""

        if self.config.get('verbose_export'):
            print(export_command)
        os.system(export_command)

    def export_model_onnx(self):

        print(f"Saving model to {self.config.get('model_name_onnx')}")
        res_path = os.path.join(os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))

        dummy_input = torch.randn(1, 1, 145, 128, 128)

        torch.onnx.export(self.model, dummy_input, res_path,
                        opset_version=11, do_constant_folding=True,
                        input_names=['input'], output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}},
                        verbose=True)

                        