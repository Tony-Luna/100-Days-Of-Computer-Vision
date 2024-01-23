import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def running_average(file_path, alpha=0.005):
    """
    Running Average Method:
    - Strengths: Effective in videos with minimal variation in the background over time.
    - Weaknesses: Not ideal for dynamic scenes; slow adaptation to changes in lighting or scene.
    """
    cap = cv2.VideoCapture(file_path)
    first_iter = True
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="Running Average Progress"):
        ret, frame = cap.read()
        if not ret:
            break
        if first_iter:
            avg = np.float32(frame)
            first_iter = False
        cv2.accumulateWeighted(frame, avg, alpha)

    result = cv2.convertScaleAbs(avg)
    cap.release()
    return result

def median_filtering(file_path, sample_count=30):
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

def mog2_background_subtraction(file_path, history=500, varThreshold=16, alpha=0.01):
    """
    MOG2 Background Subtraction Method:
    - Strengths: Adapts quickly to changes in the scene; effective for a variety of scenarios.
    - Weaknesses: May struggle with shadows and lighting variations; requires parameter tuning.
    """
    cap = cv2.VideoCapture(file_path)
    backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=False)
    avg_background = None
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        backSub.apply(frame)

        if avg_background is None:
            avg_background = frame.astype(float)
        else:
            cv2.accumulateWeighted(frame, avg_background, alpha)
        count += 1

    background_image = np.array(np.round(avg_background), dtype=np.uint8)
    cap.release()
    return background_image

def display_and_save_results(*images, titles, save_paths):
    """
    Display and save result images with enhanced visualization for professional sharing.
    """
    plt.figure(figsize=(len(images) * 6, 6))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        cv2.imwrite(save_paths[i], image)
    plt.tight_layout()
    plt.savefig("Outputs/comparison.jpg")
    plt.show()

# Main program
file_path = 'Inputs/bellas_artes_timelapse.mp4'

# Apply techniques
result1 = running_average(file_path)
result2 = median_filtering(file_path)
result3 = mog2_background_subtraction(file_path)

# Display and save results with enhanced titles
titles = [
    "Running Average: Stable Scenes",
    "Median Filtering: Noise Resilience",
    "MOG2 Subtraction: Dynamic Adaptation"
]
save_paths = [
    'Outputs/running_average_result.jpg',
    'Outputs/median_filtering_result.jpg',
    'Outputs/static_background.jpg'
]

display_and_save_results(result1, result2, result3, titles=titles, save_paths=save_paths)