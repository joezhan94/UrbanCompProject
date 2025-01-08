import os
import random
import shutil
from PIL import Image

# Set a random seed for reproducibility
random.seed(42)

# Paths to the image and mask folders
image_folder = 'preprocess/image'
mask_folder = 'preprocess/mask'
mat_folder = 'preprocess/link_key_points_final'

# Paths to save the training and test sets
train_image_folder = 'dataset/train'
train_mask_folder = 'dataset/train'
train_mat_folder = 'dataset/train'
test_image_folder = 'dataset/test/sat'
test_mask_folder = 'dataset/test/truth'
test_mat_folder = 'dataset/test/truth'

# Create output directories if they don't exist
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(test_image_folder, exist_ok=True)
os.makedirs(test_mask_folder, exist_ok=True)

# Get the list of file names (without extensions)
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]

# Ensure the files are paired and have the same base names
image_base_names = {os.path.splitext(f)[0] for f in image_files}
mask_base_names = {os.path.splitext(f)[0] for f in mask_files}
mat_base_names = {os.path.splitext(f)[0] for f in mat_files}
assert image_base_names == mask_base_names & image_base_names == mask_base_names, "Image and mask files do not match!"

# Convert the sets to lists and shuffle
paired_files = list(image_base_names)
random.shuffle(paired_files)

# Split into 80% training and 20% test
split_index = int(0.8 * len(image_files))
train_files = paired_files[:split_index]
test_files = paired_files[split_index:]

# Function to convert RGBA mask to RGB
def convert_to_rgb(mask_src, mask_dst):
    rgba_image = Image.open(mask_src).convert("RGBA")
    # Create a new black background image
    black_background = Image.new("RGB", rgba_image.size, (0, 0, 0))
    white_lines = rgba_image
    black_background.paste(white_lines, (0, 0), white_lines)  # Use the alpha channel as the mask
    # Save the resulting RGB image
    black_background.save(mask_dst)

# Function to copy files to the respective folders
def copy_files(file_list, src_image_folder, src_mask_folder, src_mat_folder, dst_image_folder, dst_mask_folder, dst_mat_folder, image_ext, mask_ext, mat_ext):
    for base_name in file_list:
        # Construct file paths
        image_src = os.path.join(src_image_folder, base_name + image_ext)
        mask_src = os.path.join(src_mask_folder, base_name + mask_ext)
        mat_src = os.path.join(src_mat_folder, base_name + mat_ext)
        image_dst = os.path.join(dst_image_folder, base_name + "_sat" + image_ext)
        mask_dst = os.path.join(dst_mask_folder, base_name + "_mask" + mask_ext)
        mat_dst = os.path.join(dst_mat_folder, base_name + "_mask" + mat_ext)

        # Copy image file
        if os.path.exists(image_src):
            shutil.copy(image_src, image_dst)
        else:
            print(f"Warning: Image file does not exist: {image_src}")

        # Copy mask file
        if os.path.exists(mask_src):
            convert_to_rgb(mask_src, mask_dst)
        else:
            print(f"Warning: Mask file does not exist: {mask_src}")

        # Copy mat file
        if os.path.exists(mat_src):
            shutil.copy(mat_src, mat_dst)
        else:
            print(f"Warning: Mat file does not exist: {mat_src}")

# Copy files to training and test folders
copy_files(train_files, image_folder, mask_folder, mat_folder, train_image_folder, train_mask_folder, train_mat_folder, '.png', '.png', '.mat')
copy_files(test_files, image_folder, mask_folder, mat_folder, test_image_folder, test_mask_folder, test_mat_folder, '.png', '.png', '.mat')

print(f"Split completed: {len(train_files)} training pairs, {len(test_files)} test pairs.")