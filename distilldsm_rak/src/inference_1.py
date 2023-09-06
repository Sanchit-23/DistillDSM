import argparse
from distilldsm_rak.src.utils.inference_utils import inference_model

def main(args):
    config = {
        'image_path' : args.image_path,
        'mask_path' : args.mask_path,
        'data_split': args.data_split,
        'gpu': args.gpu,
        'checkpoint': args.checkpoint,
        'max_samples': args.max_samples,
        'model_file': args.model_file
    }
    inference_model(config,run_type=args.run_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("""
    Inference script.
    Provide the folder paths which contains images.                                
    Provide the folder paths which contains labels.                                
    Provide the folder paths which contains data_split file.
    (Optionally) You may choose to use GPU (--gpu).
    (Optionally) You may choose to provide path to model weights, to run inference using pytorch(--model_file).
    (Optionally) You may choose to provide path to onnx or ir model, for running inference if run_type is onnx or ir(--checkpoint).
    (Optionally) You may choose to provide the max no. of samples to run inference on(--max_samples). Default value is 10.
    (Optionally) You may choose to provide run type(--run_type). Default runtype is pytorch.
    """)
    parser.add_argument('--image_path', type=str, required=True,
                        help='Folder path of images')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Folder path of labels')
    parser.add_argument('--data_split', type=str, required=True,
                        help='path to the data_split.json file')
    parser.add_argument('--gpu', type=str, help='Want GPU ?', required=False, default='False')
    parser.add_argument('--model_file', type=str, required=False,
                        help='Path of model weights saved for running inference with pytorch')
    parser.add_argument('--checkpoint', type=str,required=False,
                        help='Path of onnx model file to load for inference. Required if run type is onnx or ir')
    parser.add_argument('--max_samples', type=int, required=False, default=10,
                        help='Max no. of samples to run inference on' )
    parser.add_argument('--run_type', type=str, required=False, default='pytorch',
                        help='Chosse run type out of pytorch, onnx, ir. Default run type is pytorch')
    arguments = parser.parse_args()

    main(arguments)