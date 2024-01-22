# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:08:27 2024

@author: anlun
"""

import cv2
import os
import matplotlib.pyplot as plt

def list_jpg_files(directory):
    jpg_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            jpg_files.append(filename)
    return jpg_files

def create_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def display_and_save_subplots(images, output_directory, subplot_title):
    fig, axs = plt.subplots(1, len(images), figsize=(8, 5))
    fig.suptitle(subplot_title)

    for i, img in enumerate(images):
        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].axis('off')

    plt.savefig(os.path.join(output_directory, 'all_images_subplot.jpg'))
    plt.show()

def stitch_images(directory, output_directory):
    image_files = list_jpg_files(directory)
    images = [cv2.imread(os.path.join(directory, file)) for file in image_files]

    # Display and save the subplot of all images
    display_and_save_subplots(images, output_directory, "Original Images")

    print("Starting the stitching process. This may take a while...")
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Stitching completed. Saving the stitched image.")
        cv2.imwrite(os.path.join(output_directory, 'panorama.jpg'), stitched_image)
        plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_directory, 'stitched_image.jpg'))
        plt.show()
    else:
        print("Stitching failed. Error code: ", status)

def main(images_directory):
    output_directory = "output"
    create_output_directory(output_directory)
    stitch_images(images_directory, output_directory)
    
if __name__ == "__main__":
    images_directory = "Images Art"
    
    main(images_directory)
