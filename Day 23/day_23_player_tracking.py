from ultralytics import YOLO
import cv2
from norfair import Detection, Tracker
from tqdm import tqdm
import numpy as np

MODEL_PATH = "Models/yolov8-soccer.pt"
INPUT_VIDEO_PATH = "Videos/test_video.mp4"
OUTPUT_VIDEO_PATH = 'outputs/player_tracking.mp4'

def extract_detections_to_dict(detection_results):
    # Extracting coordinates and ids directly as in the original code
    coordinates = detection_results[0].boxes.xywh.cpu().numpy()
    ids = detection_results[0].boxes.id.int().cpu().tolist() if detection_results[0].boxes.id is not None else np.zeros(len(coordinates), dtype=int)
    
    # Creating dictionary as originally done
    detection_dict = {(round(coord[0], 2), round(coord[1], 2)): {"dimensions": (coord[2], coord[3]), "track_id": ids[i]} for i, coord in enumerate(coordinates)}
    return detection_dict

def apply_norfair_tracking(detection_results, tracker_instance):
    # Keeping the original structure for applying Norfair tracking
    initial_detections_dict = extract_detections_to_dict(detection_results)
    updated_tracking_dict = {}

    norfair_detections = [Detection(np.array(point)) for point in initial_detections_dict.keys()]
    tracked_items = tracker_instance.update(detections=norfair_detections)

    # Updating tracking dictionary based on Norfair tracking results
    for item in tracked_items:
        detection_point = tuple(item.last_detection.points[0])
        if detection_point in initial_detections_dict:
            updated_tracking_dict[detection_point] = initial_detections_dict[detection_point]
            updated_tracking_dict[detection_point]["track_id"] = item.id

    return updated_tracking_dict

def annotate_video_frames(frame, tracking_data, annotation_color):
    # Annotation process unchanged, maintaining the calculation for bounding boxes
    annotated_frame = frame.copy()
    
    for point, data in tracking_data.items():
        width, height = data['dimensions']
        object_id = data['track_id']
        
        # Calculating bounding box corners and positioning text as originally specified
        top_left = (int(point[0] - width / 2), int(point[1] - height / 2))
        bottom_right = (int(point[0] + width / 2), int(point[1] + height / 2))
        
        cv2.rectangle(annotated_frame, top_left, bottom_right, annotation_color, 2)
        
        text = str(object_id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_origin = (int(point[0] - text_size[0] / 2), top_left[1] - 10)
        
        cv2.putText(annotated_frame, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation_color, 2)

    return annotated_frame


def process_video():
    # Video processing setup and loop structure unchanged
    yolo_model = YOLO(MODEL_PATH)
    video_capture = cv2.VideoCapture(INPUT_VIDEO_PATH)
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), video_capture.get(cv2.CAP_PROP_FPS), (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_indicator = tqdm(total=frame_count, desc="Processing Video")
    
    object_tracker = Tracker(distance_function="euclidean", distance_threshold=100)
    
    while video_capture.isOpened():
        read_success, frame = video_capture.read()
        if not read_success:
            break
        
        detection_results = yolo_model(frame, classes=[1, 3])
        tracking_results = apply_norfair_tracking(detection_results, object_tracker)
        frame_with_annotations = annotate_video_frames(frame, tracking_results, (0, 255, 255))
        
        video_writer.write(frame_with_annotations)
        progress_indicator.update(1)
    
    video_capture.release()
    video_writer.release()
    progress_indicator.close()

if __name__ == "__main__":
    process_video()
