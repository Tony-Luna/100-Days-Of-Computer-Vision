# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 00:20:35 2024

@author: anlun
"""

from skimage.transform import resize
import numpy as np
import cv2
from matplotlib import pyplot as plt

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

def linear_normalization(array, new_min=0, new_max=1):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    normalized_array = normalized_array * (new_max - new_min) + new_min
    return normalized_array

# Compute the activation of the input map using the Markovian approach shown in 
# the GBVS algorithm -----------------------------------------------------------
def markov_activation(maps_list):
    markov_maps = []
    
    for img in maps_list:
        img = resize(img, (32,32))  # Reduce image size for Markovian activation
        n = img.size
        
        # Compute distance matrix
        ix, iy = np.indices(img.shape)
        ix = ix.reshape(n, 1)
        iy = iy.reshape(n, 1)
        d = (ix - ix.T)**2 + (iy - iy.T)**2
        # Generate weight matrix between nodes based on distance matrix
        sig = (0.15) * np.mean(img.shape)
        Dw = np.exp(-1 * d / (2 * sig**2))
        
        # Assign a linear index to each node
        linear_map = img.ravel()
        
        # Assign edge weights based on distances between nodes and algtype
        MM = Dw * np.abs(linear_map[:, None] - linear_map)
        
        # Make it a markov matrix (so each column sums to 1)
        MM = safe_divide(MM, np.sum(MM, axis=0, keepdims=True))
        
        # Find the principal eigenvector using matrix_power function
        v = np.ones((n, 1), dtype=np.float32) / n
        MM_pow = np.linalg.matrix_power(MM, 5)
        Vo = MM_pow @ v
        Vo = safe_divide(Vo, np.sum(Vo))
        
        # Arrange the nodes back into a rectangular map
        activation = Vo.reshape(img.shape)
        markov_maps.append(activation)
        
    combined_map = linear_normalization(np.sum(markov_maps, axis=0))
    combined_map = resize(combined_map, maps_list[0].shape)
    
    return combined_map

def main(image_paths):
    num_images = len(image_paths)
    plt.figure(figsize=(15, num_images * 5))  # Adjust the size as needed

    for i, image_path in enumerate(image_paths):
        # Load image using OpenCV
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Unable to load image at {image_path}")
            continue

        # Convert the image to grayscale and normalize
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_img_normalized = gray_img / 255.0

        # Pass the grayscale converted image through the markov_activation
        activation_map = markov_activation([gray_img_normalized])

        # Plotting original image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image {i+1}', fontsize=12)
        plt.axis('off')

        # Plotting activation map
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(activation_map)
        plt.title(f'Activation Map {i+1}', fontsize=12)
        plt.axis('off')

    plt.suptitle('Image and Activation Map Comparisons', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to fit the super title
    plt.show()

# Example usage
image_paths = ["Images/1.jpg", "Images/2.jpg", "Images/3.jpg"]  # List your image paths here
main(image_paths)
