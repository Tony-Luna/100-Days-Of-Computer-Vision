# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 00:20:35 2024
@author: anlun
"""

import numpy as np
import cv2
from skimage.transform import resize
from matplotlib import pyplot as plt

# Constants
IMAGE_SIZE = (32, 32)
SIGMA_FACTOR = 0.15
MATRIX_POWER = 5
FIG_SIZE = (15, 5)

def safe_divide(a, b):
    """ Safely divide two arrays element-wise. """
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

def linear_normalization(array, new_min=0, new_max=1):
    """ Normalize an array to a new range. """
    min_val, max_val = np.min(array), np.max(array)
    return ((array - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

def compute_distance_matrix(shape):
    """ Compute the distance matrix for a given shape. """
    n = np.prod(shape)
    ix, iy = np.indices(shape)
    d = (ix.reshape(n, 1) - ix.ravel())**2 + (iy.reshape(n, 1) - iy.ravel())**2
    return d, n

def markov_activation(maps_list):
    """ Compute the activation of the input map using a Markovian approach. """
    markov_maps = []

    for img in maps_list:
        img = resize(img, IMAGE_SIZE)
        d, n = compute_distance_matrix(img.shape)
        sig = SIGMA_FACTOR * np.mean(img.shape)
        Dw = np.exp(-1 * d / (2 * sig**2))
        linear_map = img.ravel()
        MM = Dw * np.abs(linear_map[:, None] - linear_map)
        MM = safe_divide(MM, np.sum(MM, axis=0, keepdims=True))
        v = np.ones((n, 1), dtype=np.float32) / n
        Vo = np.linalg.matrix_power(MM, MATRIX_POWER) @ v
        Vo = safe_divide(Vo, np.sum(Vo))
        activation = Vo.reshape(img.shape)
        markov_maps.append(activation)

    combined_map = linear_normalization(np.sum(markov_maps, axis=0))
    return resize(combined_map, maps_list[0].shape)

def plot_images(image_paths):
    """ Plot original images and their activation maps. """
    num_images = len(image_paths)
    plt.figure(figsize=(FIG_SIZE[0], num_images * FIG_SIZE[1]))

    for i, image_path in enumerate(image_paths):
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Unable to load image at {image_path}")
            continue

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) / 255.0
        activation_map = markov_activation([gray_img])

        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image {i+1}')
        plt.axis('off')

        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(activation_map)
        plt.title(f'Activation Map {i+1}')
        plt.axis('off')

    plt.suptitle('Graph-Based Visual Saliency (Activation)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def main():
    """ Main function to process the images. """
    image_paths = ["Images/1.jpg", "Images/2.jpg", "Images/3.jpg"]
    plot_images(image_paths)

if __name__ == "__main__":
    main()
