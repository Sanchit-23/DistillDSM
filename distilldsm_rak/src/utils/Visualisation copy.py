import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.utils.distill_dsm import U_Net
import numpy as np

def load_model_weights(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

def visualize_segmentation(input_image, ground_truth):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Original Image
    axs[0].plt.imshow(input_image)
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    # Ground Truth Mask
    axs[1].imshow(ground_truth, cmap='gray')
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')

    plt.show()


def main():
    model = U_Net(1, 1, conv_type='conv_2d', tsm=True, tsm_length=145, learn=True)

    # Load pre-trained model weights
    model_weights_path = 'model_weights/model_weights_best_seg.pt'
    load_model_weights(model, model_weights_path)

    # Load the input image and ground truth mask for the first sample in the training dataset
    input_image_path = 'dataset/New_Dataset/img_volumes/PANCREAS_0001.npy'
    ground_truth_path = 'dataset/New_Dataset/labels/PANCREAS_0001.npy'

    input_image = np.load(input_image_path)
    ground_truth = np.load(ground_truth_path)




    # Visualize the input image, ground truth, and segmentation output
    visualize_segmentation(input_image,
                           ground_truth,
                          )

if __name__ == '__main__':
    main()