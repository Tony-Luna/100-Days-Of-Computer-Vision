import cv2
import numpy as np
import os

class FaceFilter:
    def __init__(self, model_path, ninja_mask_path, input_size=(320, 320), conf_threshold=0.9, nms_threshold=0.3, top_k=5000):
        self.face_detector = cv2.FaceDetectorYN.create(model_path, "", input_size, conf_threshold, nms_threshold, top_k)
        self.ninja_mask = cv2.imread(ninja_mask_path, cv2.IMREAD_UNCHANGED)
        self.ninja_mask = self.scale_image(self.ninja_mask)
        
        # Hardcoded normalized eye locations in the mask
        self.ninja_eye_left = (0.35, 0.9)
        self.ninja_eye_right = (0.65, 0.9)

    def scale_image(self, image, max_size=640):
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
        return image

    def pad_image(self, image):
        height, width = image.shape[:2]
        max_side = max(height, width)
        padded_image = cv2.copyMakeBorder(image, 
                                          top=(max_side - height) // 2, 
                                          bottom=(max_side - height + 1) // 2, 
                                          left=(max_side - width) // 2, 
                                          right=(max_side - width + 1) // 2, 
                                          borderType=cv2.BORDER_CONSTANT, 
                                          value=[0, 0, 0, 0])
        return padded_image

    def apply_mask(self, image, mask, eye_left, eye_right):
        mask_width = int(np.linalg.norm(eye_left - eye_right) * 2.5)
        mask_height = int(mask_width * mask.shape[0] / mask.shape[1])
        mask_resized = cv2.resize(mask, (mask_width, mask_height), interpolation=cv2.INTER_AREA)

        eye_center = (eye_left + eye_right) // 2
        mask_eye_left = np.array(self.ninja_eye_left) * [mask_resized.shape[1], mask_resized.shape[0]]
        mask_eye_right = np.array(self.ninja_eye_right) * [mask_resized.shape[1], mask_resized.shape[0]]
        
        # Calculate the angle to rotate the mask
        angle = np.degrees(np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))
        
        # Pad the mask to avoid cropping during rotation
        mask_padded = self.pad_image(mask_resized)
        mask_center = (mask_padded.shape[1] // 2, mask_padded.shape[0] // 2)
        M = cv2.getRotationMatrix2D(mask_center, -angle, 1)
        mask_rotated = cv2.warpAffine(mask_padded, M, (mask_padded.shape[1], mask_padded.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        mask_eye_center = (mask_eye_left + mask_eye_right) // 2
        top_left_x = int(eye_center[0] - mask_eye_center[0])
        top_left_y = int(eye_center[1] - mask_eye_center[1])

        alpha_mask = mask_rotated[:, :, 3] / 255.0
        for i in range(mask_rotated.shape[0]):
            for j in range(mask_rotated.shape[1]):
                y_offset = int(top_left_y + i - mask_padded.shape[0] // 2 + mask_eye_center[1])
                x_offset = int(top_left_x + j - mask_padded.shape[1] // 2 + mask_eye_center[0])
                if 0 <= y_offset < image.shape[0] and 0 <= x_offset < image.shape[1]:
                    image[y_offset, x_offset] = image[y_offset, x_offset] * (1 - alpha_mask[i, j]) + mask_rotated[i, j, :3] * alpha_mask[i, j]
        return image

    def draw_tech_corners(self, image, bbox, color=(255, 0, 0), thickness=2, length=20):
        x, y, w, h = bbox
        # Top-left corner
        cv2.line(image, (x, y), (x + length, y), color, thickness)
        cv2.line(image, (x, y), (x, y + length), color, thickness)
        # Top-right corner
        cv2.line(image, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(image, (x + w, y), (x + w, y + length), color, thickness)
        # Bottom-left corner
        cv2.line(image, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(image, (x, y + h), (x, y + h - length), color, thickness)
        # Bottom-right corner
        cv2.line(image, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def visualize(self, image, faces, fps=None):
        output = image.copy()
        if fps is not None:
            cv2.putText(output, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for face in faces:
            bbox = face[:4].astype(int)
            score = face[-1]
            landmarks = face[4:14].reshape(5, 2).astype(int)
            eye_left, eye_right = landmarks[0], landmarks[1]

            # Draw tech corners instead of a full bounding box
            self.draw_tech_corners(output, bbox, color=(255, 0, 0), thickness=2, length=20)
            cv2.putText(output, f'Score: {score:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw landmarks
            for landmark in landmarks:
                cv2.circle(output, tuple(landmark), 2, (0, 255, 0), 2)
            
            # Apply ninja mask to the face using eye landmarks
            output = self.apply_mask(output, self.ninja_mask, eye_left, eye_right)

        return output

    def run(self, frame):
        self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.face_detector.detect(frame)
        return faces

def main():
    model_path = 'models/face_detection_yunet_2023mar.onnx'
    ninja_mask_path = 'data/glasses.png'
    face_filter = FaceFilter(model_path, ninja_mask_path)

    # Ensure the output directory exists
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output_video.mp4')

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tm = cv2.TickMeter()
        tm.start()
        faces = face_filter.run(frame)
        tm.stop()

        if faces is not None:
            frame = face_filter.visualize(frame, faces, fps=tm.getFPS())

        # Write the frame to the output video file
        out.write(frame)

        cv2.imshow('Face Detection with Ninja Mask', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

