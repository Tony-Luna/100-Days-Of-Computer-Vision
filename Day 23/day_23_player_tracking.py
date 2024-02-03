# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 03:41:03 2024

@author: anlun
"""

from ultralytics import YOLO
import cv2
from norfair import Detection, Tracker
from tqdm import tqdm
import numpy as np

YOLO_MODEL = "Models/yolov8-soccer.pt"
VIDEO_PATH = "Videos/test_video.mp4"
OUTPUT_VIDEO_PATH = 'outputs/player_tracking.mp4'

def yolo_pred_to_dict(yolo_results):
    """
    Converts YOLO predictions to a dictionary format for easy processing.
    Each bounding box is associated with its width, height, and track ID.
    """
    # Extract coordinates and track IDs
    xywh_coord = yolo_results[0].boxes.xywh.cpu().numpy()
    track_id = yolo_results[0].boxes.id.int().cpu().tolist() if yolo_results[0].boxes.id is not None else np.zeros(xywh_coord.shape[0], dtype=int)
    
    # Create dictionary with coordinates as keys and width, height, and ID as values
    dict_xy_wh_id = {(round(row[0], 2), round(row[1], 2)): {"wh": (row[2], row[3]), "id":track_id[i]} for i, row in enumerate(xywh_coord)}
    return dict_xy_wh_id

def get_norfair_track_results_dict(yolo_results, tracker):
    """
    Applies Norfair tracking on YOLO predictions and returns a dictionary of results.
    """
    dict_results = yolo_pred_to_dict(yolo_results)
    dict_results_norfair = {}

    # Convert detections to Norfair format and update tracker
    norfair_detections = [Detection(np.array(points)) for points in dict_results.keys()]
    tracked_objects = tracker.update(detections=norfair_detections)

    # Update tracking IDs in the results dictionary
    for tracked_obj in tracked_objects:
        last_point = tuple(tracked_obj.last_detection.points[0])
        
        if last_point in dict_results:
            dict_results_norfair[last_point] = dict_results[last_point]
            dict_results_norfair[last_point]["id"] = tracked_obj.id

    return dict_results_norfair

def draw_game_style_annotations(original_frame, annotations_dictionary, circle_color):
    """
    Draws annotations on the video frame indicating tracked objects with a game-style visualization.
    Each bounding box has a downward-pointing triangle and a semi-ellipse at its bottom.
    """
    frame = original_frame.copy()
    
    # Fixed dimensions for the triangle
    tri_height = 25  # Fixed height of the triangle
    tri_base_half_length = 15  # Half the base length of the triangle
    vertical_offset = 20  # Vertical offset for the triangle above the bounding box

    for center, info in annotations_dictionary.items():
        width, height = info['wh']
        box_id = info['id']
        
        # Calculate the bottom center of the bounding box for the ellipse
        bottom_center = (int(center[0]), int(center[1] + height / 2))
        
        # Define the axes for the ellipse and angle
        ellipse_axes = (int(width / 2), int(height / 10))  # Further reduced height for the semi-ellipse
        ellipse_angle = 0  # Angle of the ellipse
        ellipse_thickness = 4  # Reduced thickness of the ellipse for a neater appearance

        # Draw the bottom half ellipse in blue with the specified thickness
        cv2.ellipse(frame, bottom_center, ellipse_axes, ellipse_angle, 0, 180, circle_color, ellipse_thickness)

        # Calculate the bottom point of the triangle (above the bounding box)
        top_point_triangle = (int(center[0]), int(center[1] - height / 2) - vertical_offset)

        # Triangle points
        p1 = (top_point_triangle[0], top_point_triangle[1] + tri_height)  # Bottom point of the triangle
        p2 = (top_point_triangle[0] - tri_base_half_length, top_point_triangle[1])  # Top-left point
        p3 = (top_point_triangle[0] + tri_base_half_length, top_point_triangle[1])  # Top-right point

        # Draw the filled triangle in white for the ID
        cv2.drawContours(frame, [np.array([p1, p2, p3])], 0, (255, 255, 255), -1)

        # Add the ID text in black, centered in the triangle
        text_size = cv2.getTextSize(str(box_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = p1[0] - text_size[0] // 2
        text_y = p1[1] - 2* text_size[1] // 3  # Adjusted for centering text in the triangle
        cv2.putText(frame, str(box_id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame

def main():
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    
    # Determine the number of frames in the video (for the progress bar)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing Video")
    
    # Initialize norfair tracker
    tracker = Tracker(distance_function="euclidean", distance_threshold=100)
    
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Track objects in the frame using norfair
            results = model(frame, classes=[1,3])
    
            # Convert results to dictionary of points-dimensions
            dict_results_norfair = get_norfair_track_results_dict(results, tracker)
    
            # Visualize the detection and tracking results on the frame
            annotated_frame_norfair = draw_game_style_annotations(frame, dict_results_norfair, circle_color=(255, 0, 0))
            
            # Write the frame to the output video
            out_video_writer.write(annotated_frame_norfair)
            
            # Update the progress bar
            progress_bar.update(1)
        else:
            # Break the loop if the end of the video is reached
            break
    
    # Finalize the progress bar
    progress_bar.close()
    
    # Release the video capture and writer object, then close all the frames
    cap.release()
    out_video_writer.release()
    
if __name__ == "__main__":
    main()