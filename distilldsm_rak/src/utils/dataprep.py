import os
import numpy as np
# Set the path to the main folder
main_folder = '/home/sanchit/bmi7_new_implentation/dataset/pancreas_data_volumes'

# Set the paths to the images and masks folders
images_folder = os.path.join(main_folder, 'img_volumes')
masks_folder = os.path.join(main_folder, 'labels')

# Get a list of image files in the images folder
image_files = os.listdir(images_folder)

# Iterate over each image file
for i, image_file in enumerate(image_files):
    # Load the image and mask
    image_path = os.path.join(images_folder, image_file)
    mask_path = os.path.join(masks_folder, image_file)
    image = np.load(image_path)
    mask = np.load(mask_path)

    # Find the channels with a non-zero mask value
    non_zero_channels = np.any(mask, axis=(0, 1))
    
    # Extract the channels with non-zero mask values
    new_image = image[:, :, non_zero_channels]
    new_mask = mask[:, :, non_zero_channels]

    #Set the file names for the image and mask
    image_filename = f"PANCREAS_{i+1:04d}.npy"
    mask_filename = f"PANCREAS_{i+1:04d}.npy"

    # Save the image and mask in the respective subfolders
    np.save(os.path.join('/home/sanchit/bmi7_new_implentation/dataset/New_Dataset/img_volumes',
                          image_filename), new_image)
    np.save(os.path.join('/home/sanchit/bmi7_new_implentation/dataset/New_Dataset/labels',
                          mask_filename), new_mask)

print("New dataset saved successfully.")
