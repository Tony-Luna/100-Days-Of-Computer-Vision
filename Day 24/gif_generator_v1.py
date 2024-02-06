# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 00:56:44 2024

@author: anlun
"""

from PIL import Image
import os
from tqdm import tqdm

def create_gif_with_uniform_duration_and_size(image_folder, output_path, frame_duration):
    """
    Create a GIF from images in a specified folder with a uniform frame duration and uniform frame size.

    :param image_folder: Path to the folder containing the images.
    :param output_path: Path where the GIF should be saved.
    :param frame_duration: Duration for each frame in milliseconds.
    """
    # List images in the folder
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(images)
    
    # Open and append each image to a list
    frames = []
    for i, image_name in tqdm(enumerate(images), desc="Processing frames", total=len(images)):
        with Image.open(os.path.join(image_folder, image_name)) as img:
            if i == 0:
                first_image_size = img.size  # Capture the size of the first image
            frames.append(img.resize(first_image_size, Image.LANCZOS).copy())  # Resize to match the first image

    # Save the frames as a GIF with a uniform frame duration
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)


# Example usage
image_folder_path = 'C:/Users/anlun/OneDrive/Documents/GitHub/Im2Oil/output/test_7-p-4/process'
output_gif_path = 'outputs/output_7.gif'
frame_duration = 300  # Uniform frame duration in milliseconds for all images

create_gif_with_uniform_duration_and_size(image_folder_path, output_gif_path, frame_duration)
