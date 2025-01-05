import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def get_random_file_for_city(folder, city):
    """
    Returns a random file from the folder that starts with the city name.
    """
    # Get all files that start with the city name
    files = [f for f in os.listdir(folder) if f.startswith(city)]

    # Randomly select one file
    return random.choice(files)

def generate_image_grid(source, truth_folder, target, cities, output):
    """
    Generates a 6x3 image grid showing the satellite image, predicted mask, and ground truth mask for each city.
    """
    # Initialize the grid
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, city in enumerate(cities):
        # Randomly select a satellite image for the city
        sat_file = get_random_file_for_city(source, city)
        sat_image_path = os.path.join(source, sat_file)
        sat_image = cv2.imread(sat_image_path)
        sat_image = cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB) 

        # Get the corresponding predicted mask
        pred_mask_file = sat_file.replace("_sat.png", "_mask.png")
        pred_mask_path = os.path.join(target, pred_mask_file)
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

        # Get the corresponding ground truth mask
        truth_mask_file = sat_file.replace("_sat.png", "_mask.png")
        truth_mask_path = os.path.join(truth_folder, truth_mask_file)
        truth_mask = cv2.imread(truth_mask_path, cv2.IMREAD_GRAYSCALE)

        # Plot the satellite image
        axes[i, 0].imshow(sat_image)
        axes[i, 0].set_title(f"{city.capitalize()} Satellite Image")
        axes[i, 0].axis("off")

        # Plot the predicted mask
        axes[i, 1].imshow(pred_mask, cmap="gray")
        axes[i, 1].set_title(f"{city.capitalize()} Predicted Mask")
        axes[i, 1].axis("off")

        # Plot the ground truth mask
        axes[i, 2].imshow(truth_mask, cmap="gray")
        axes[i, 2].set_title(f"{city.capitalize()} Ground Truth Mask")
        axes[i, 2].axis("off")

    plt.savefig(os.path.join(output, "image_grid.png"), bbox_inches="tight")
    plt.close()
    print("Image grid saved successfully!")

if __name__ == "__main__":

    source = 'dataset/test/sat/'  
    truth_folder = 'dataset/test/truth/'  
    target = 'dataset/test/pred/mask/' 
    output = 'fig/'

    # List of cities
    cities = ["amsterdam", "copenhagen", "paris", "london", "strasbourg", "utrecht"]

    # Generate the image grid
    generate_image_grid(source, truth_folder, target, cities, output)