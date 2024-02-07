# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:42:40 2024

@author: anlun
"""

import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    # Access global variables
    global points, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:  # Limit to 4 points
            points.append((x, y))
            # Draw the clicked point with its number
            cv2.circle(current_image, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(current_image, str(len(points)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("image", current_image)
        if len(points) == 4:
            # Prompt to keep or reset points
            cv2.putText(current_image, "Keep points? (y/n)", (50, current_image.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("image", current_image)

def select_points(image):
    global points, current_image
    points = []  # Reset points for each call
    current_image = image.copy()
    cv2.imshow('image', current_image)
    cv2.setMouseCallback('image', click_event)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y'):
            cv2.destroyAllWindows()
            return points
        elif key == ord('n'):
            points = []
            current_image = image.copy()
            cv2.imshow('image', current_image)

def apply_homography(test_img, template_img, test_pts, template_pts):
    H, _ = cv2.findHomography(np.array(test_pts), np.array(template_pts))
    height, width, channels = template_img.shape
    transformed_img = cv2.warpPerspective(test_img, H, (width, height))
    return transformed_img

def main(config):
    global current_image  # Needed for displaying the images
    
    # Load images
    test_image = cv2.imread(config['test_image_path'], cv2.IMREAD_COLOR)
    template_image = cv2.imread(config['template_image_path'], cv2.IMREAD_COLOR)
    
    # Select points on the test image
    print("Select 4 points on the test image, then press 'y' to confirm or 'n' to restart.")
    test_points = select_points(test_image)
    
    # Select points on the template image
    print("Select 4 points on the template image, then press 'y' to confirm or 'n' to restart.")
    template_points = select_points(template_image)
    
    # Apply homography
    transformed_image = apply_homography(test_image, template_image, test_points, template_points)
    
    # Display and save the transformed image
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optional: Save the transformed image to a file
    cv2.imwrite("transformed_image.png", transformed_image)

if __name__ == "__main__":
    config = {
        'test_image_path': 'Images/test_2.png',
        'template_image_path': 'Images/field_template.png'
    }
    
    main(config)
