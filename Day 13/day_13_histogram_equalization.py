import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(img, title):
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title(title)

def equalize_hist_color(img):
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    return cv2.merge(eq_channels)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(clahe.apply(ch))
    return cv2.merge(eq_channels)

def main():
    # Load the image
    img = cv2.imread('Images/4.jpg')
    
    # Color histogram equalization
    eq_color_img = equalize_hist_color(img)
    
    # CLAHE on color image
    clahe_img = apply_clahe(img)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Original Image
    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Original Histogram
    plt.subplot(3, 2, 2)
    plot_histogram(img, 'Original Histogram')
    
    # Equalized Image
    plt.subplot(3, 2, 3)
    plt.imshow(cv2.cvtColor(eq_color_img, cv2.COLOR_BGR2RGB))
    plt.title('Equalized Image')
    plt.axis('off')
    
    # Equalized Histogram
    plt.subplot(3, 2, 4)
    plot_histogram(eq_color_img, 'Equalized Histogram')
    
    # CLAHE Image
    plt.subplot(3, 2, 5)
    plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    plt.title('CLAHE Image')
    plt.axis('off')
    
    # CLAHE Histogram
    plt.subplot(3, 2, 6)
    plot_histogram(clahe_img, 'CLAHE Histogram')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()