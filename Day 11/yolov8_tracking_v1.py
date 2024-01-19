# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:41:58 2024

@author: anlun
"""
from ultralytics import YOLO
import cv2

MODEL_PATH = 'yolov8n.pt'
VIDEO_PATH = 'Datasets/Test videos/test_2.mp4'
OUTPUT_VIDEO_PATH = 'output_video.mp4'  # Define the output video path

def load_model(model_path):
    """
    Load the YOLO model.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_video(video_path, model):
    """
    Process the video, perform object detection and tracking, display results, and save the output to a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize the Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used for MP4 format
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        frame_ = results[0].plot()

        # Write frame to video
        out.write(frame_)

        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the Video Writer
    cv2.destroyAllWindows()

def main():
    model = load_model(MODEL_PATH)
    if model:
        process_video(VIDEO_PATH, model)

if __name__ == "__main__":
    main()
