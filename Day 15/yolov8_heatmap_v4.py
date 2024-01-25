# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 02:19:36 2024

@author: anlun
"""

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os

# Constants and parameters
VIDEO_PATH = "Inputs/art_expo_timelapse.mp4"
TRACK_COLOR = (0, 255, 0)
MODEL_PATH = 'yolov8m.pt'
CIRCLE_DIAMETER_PROPORTION = 0.3  # Proportion of the width of bounding boxes
EXPERIMENT_NAME = 'Art_Expo_Analysis'  # Define the experiment name
OUTPUT_DIR = f"Outputs/{EXPERIMENT_NAME}/Heatmaps"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def median_filtering(file_path, sample_count=100):
    """
    Median Filtering Method:
    - Strengths: Robust against random noise and short-term changes in the scene.
    - Weaknesses: Requires a significant number of frames for accuracy; not suitable for real-time applications.
    """
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FOI = frame_count * np.random.uniform(size=sample_count)
    frames = []

    for frameOI in tqdm(FOI, desc="Median Filtering Progress"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    result = np.median(frames, axis=0).astype(dtype=np.uint8)
    cap.release()
    return result

def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_frame(frame, model, heatmap):
    results = model.track(frame, persist=True, classes=0)
    boxes = results[0].boxes.xywh.cpu()
    annotated_frame = results[0].plot()

    for box in boxes:
        x, y, w, _ = box  # x, y are center points, w is width of the box
        circle_radius = int(w * CIRCLE_DIAMETER_PROPORTION / 2)

        # Draw filled tracking circle on the annotated frame
        cv2.circle(annotated_frame, (int(x), int(y)), circle_radius, TRACK_COLOR, thickness=-1)

        # Update heatmap by incrementing the pixels inside the tracking circle
        mask = np.zeros_like(heatmap)
        cv2.circle(mask, (int(x), int(y)), circle_radius, (1), thickness=-1)
        heatmap += mask

    return annotated_frame

def logarithmic_normalization(heatmap):
    # Adding 1 to avoid log(0)
    log_norm_heatmap = np.log1p(heatmap) 
    max_val = np.max(log_norm_heatmap)
    if max_val > 0:
        log_norm_heatmap /= max_val  # Scale to [0, 1]
    return log_norm_heatmap

def main():
    # Background extraction
    extracted_background = median_filtering(VIDEO_PATH)
    cv2.imwrite(f"{OUTPUT_DIR}/art_background.png", extracted_background)

    # People tracking and video processing
    model = load_model(MODEL_PATH)
    if model is None:
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    # Prepare video writers
    frame_size = (frame.shape[1], frame.shape[0])
    annotated_video = cv2.VideoWriter(f'{OUTPUT_DIR}/annotated_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)
    overlay_video = cv2.VideoWriter(f'{OUTPUT_DIR}/overlay_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)

    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        annotated_frame = process_frame(frame, model, heatmap)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        annotated_video.write(annotated_frame)

        # Normalize and visualize heatmap
        heatmap_normalized = logarithmic_normalization(heatmap)
        heatmap_color_mapped = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Overlay heatmap on the current frame
        heatmap_opacity = 0.5
        overlay_frame = cv2.addWeighted(frame, 1, heatmap_color_mapped, heatmap_opacity, 0)
        cv2.imshow("Overlay Heatmap", overlay_frame)
        overlay_video.write(overlay_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    annotated_video.release()
    overlay_video.release()
    cv2.destroyAllWindows()

    # Overlay final heatmap on extracted background
    final_heatmap_overlay = cv2.addWeighted(extracted_background, 1, heatmap_color_mapped, heatmap_opacity, 0)
    cv2.imwrite(f"{OUTPUT_DIR}/final_heatmap_overlay.png", final_heatmap_overlay)

if __name__ == "__main__":
    main()