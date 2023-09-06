import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.utils.distill_dsm import U_Net
import numpy as np

def load_model_weights(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

def visualize_segmentation(input_image, ground_truth, segmentation_output):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axs[0].imshow(input_image)
    axs[0].set_title('Input Image', cmap='gray')
    axs[0].axis('off')

    # Ground Truth Mask
    axs[1].imshow(ground_truth, cmap='gray')
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')

    # Segmentation Output
    axs[2].imshow(segmentation_output, cmap='gray')
    axs[2].set_title('Segmentation Output')
    axs[2].axis('off')

    plt.show()


def main():
    model = U_Net(1, 1, conv_type='conv_2d', tsm=True, tsm_length=145, learn=True)

    # Load pre-trained model weights
    model_weights_path = 'model_weights/model_weights_best_seg.pt'
    load_model_weights(model, model_weights_path)

    # Load the input image and ground truth mask for the first sample in the training dataset
    input_image_path = 'dataset/New_Dataset/img_volumes/PANCREAS_0001.npy'
    ground_truth_path = 'dataset/New_Dataset/labels/PANCREAS_0001.npy'

    input_3d_image = np.load(input_image_path)
    ground_truth_3d = np.load(ground_truth_path)

    # Extract the 112th channel from the 3D image and convert it to a grayscale image
    input_image = input_3d_image[:, :,111]  # Extract the 112th channel


    # Extract the 112th channel from the 3D ground truth and convert it to a grayscale image
    ground_truth = ground_truth_3d[:, :, 111]  # Extract the 112th channel


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((145, 128)),  # Resize to the desired size
        transforms.ToTensor(),
    ])

    # Preprocess the input image and ground truth mask and add the batch dimension
    input = transform(input_image).unsqueeze(0)
    gt = transform(ground_truth).unsqueeze(0)

    # Perform inference using the loaded model
    with torch.no_grad():
        output = model(input)

    # Convert the output tensor to a numpy array and squeeze the batch dimension
    segmentation_output = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()

    # Visualize the input image, ground truth, and segmentation output
    visualize_segmentation(input.squeeze[0],
                           gt[0],
                           segmentation_output)

if __name__ == '__main__':
    main()