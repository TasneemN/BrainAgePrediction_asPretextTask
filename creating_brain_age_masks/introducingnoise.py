import os
import numpy as np
import nibabel as nib
from config import *

# Function to add random noise to each voxel in the image
def add_noise_to_image(image):
    noise = np.random.randint(-2, 3, size=image.shape)  # Generate random noise
    noisy_image = image + noise
    return noisy_image

# Directory containing the original images with added noise from the previous step
image_directory_age = AGE_IMAGES_DIRECTORY

# Output directory for images with additional noise
noisy_images_directory = NOISY_IMAGES_DIRECTORY


# Create output directory if it doesn't exist
os.makedirs(noisy_images_directory, exist_ok=True)

# Process each file in the image directory
for file in os.listdir(image_directory_age):
    if file.endswith(".nii.gz"):  # Use the original file names
        file_path = os.path.join(image_directory_age, file)
        print("Processing:", file_path)
        
        # Load the image
        image = nib.load(file_path).get_fdata()
        
        # Add noise to each voxel
        image_with_additional_noise = add_noise_to_image(image)
        
        # Save the image with added noise
        output_file_path = os.path.join(noisy_images_directory, file)
        nib.save(nib.Nifti1Image(image_with_additional_noise, np.eye(4)), output_file_path)
        print("Saved with additional noise:", output_file_path)
