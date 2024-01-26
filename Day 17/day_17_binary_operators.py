# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:28:23 2024

@author: anlun
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
color_img = cv2.imread('Images/tesla.jpg')
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Convert to grayscale
gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)

# Apply Otsu's thresholding to create a binary image
_, otsu_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Define a 5x5 kernel
kernel = np.ones((9, 9), np.uint8)

# Perform morphological operations
dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)
opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)

# Compute distance transform
dist_transform = cv2.distanceTransform(otsu_thresh, cv2.DIST_L2, 5)

# Find connected components
num_labels, labels = cv2.connectedComponents(otsu_thresh)

# Display all images in a single plot
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# Original Color Image
axes[0].imshow(color_img)
axes[0].set_title('Original Color Image')
axes[0].axis('off')

# Grayscale Image
axes[1].imshow(gray_img, cmap='gray')
axes[1].set_title('Grayscale Image')
axes[1].axis('off')

# Otsu's Thresholding
axes[2].imshow(otsu_thresh, cmap='gray')
axes[2].set_title("Otsu's Threshold")
axes[2].axis('off')

# Morphological Operations
axes[3].imshow(dilation, cmap='gray')
axes[3].set_title('Dilation')
axes[3].axis('off')

axes[4].imshow(erosion, cmap='gray')
axes[4].set_title('Erosion')
axes[4].axis('off')

axes[5].imshow(opening, cmap='gray')
axes[5].set_title('Opening')
axes[5].axis('off')

axes[6].imshow(closing, cmap='gray')
axes[6].set_title('Closing')
axes[6].axis('off')

# Distance Transform
axes[7].imshow(dist_transform, cmap='gray')
axes[7].set_title('Distance Transform')
axes[7].axis('off')

# Connected Components
axes[8].imshow(labels, cmap='nipy_spectral')
axes[8].set_title('Connected Components')
axes[8].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
