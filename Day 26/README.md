# 😎 Face Detection with Glasses Overlay

![Demo](assets/demo.gif)

## 📝 Overview
This project demonstrates a face detection application that overlays glasses on detected faces in real-time using OpenCV. The primary components of the project include face detection, glasses overlay application, and video processing. The output video is saved to a specified directory, ensuring the real-time feed and the saved video maintain consistent speeds.

## 🔧 Main Components

### 🖥️ FaceFilter Class
The `FaceFilter` class encapsulates the core functionality of the application. It includes methods for initializing the face detection model, scaling images, padding images, applying the glasses overlay, drawing tech-style corners, visualizing the results, and running the detection on video frames.

- **Initialization**
  - **Model Path:** Path to the face detection model.
  - **Glasses Image Path:** Path to the glasses image.
  - **Input Size:** Size of the input image for the face detector.
  - **Confidence Threshold:** Minimum confidence for a face to be considered valid.
  - **NMS Threshold:** Non-maximum suppression threshold.
  - **Top K:** Maximum number of faces to detect.

- **Scale Image**
  - Scales the input image to a specified maximum size while maintaining the aspect ratio.

- **Pad Image**
  - Pads the image to make it square, adding borders with a specified color.

- **Apply Overlay**
  - Applies the glasses overlay to a detected face using the positions of the eyes. The overlay is resized and rotated to align with the face.

- **Draw Tech Corners**
  - Draws partial corners (tech-style) around the detected face instead of a complete bounding box for a high-tech appearance.

- **Visualize**
  - Combines the detected faces, scores, landmarks, and FPS information onto the output image. It also applies the glasses overlay to each detected face.

- **Run**
  - Detects faces in the provided video frame using the initialized face detection model.

### 🎥 Main Function
The `main` function handles the video capture, processing, and saving the output video. It ensures the output directory exists and saves the processed video in real-time.

## 📈 Workflow
1. **Initialization:**
   - The `FaceFilter` class is instantiated with paths to the face detection model and glasses image.
   
2. **Video Capture:**
   - A video capture object is created to read frames from the webcam.
   
3. **Frame Processing:**
   - For each frame:
     - The frame is read from the video capture object.
     - Face detection is performed.
     - Detected faces are visualized with tech-style corners and a glasses overlay.
     - FPS is calculated and displayed.
     - The processed frame is shown in a window and saved to the output video file.
     
4. **Video Saving:**
   - The processed video frames are saved to a specified directory (`outputs`), ensuring the real-time feed and the saved video have consistent speeds.

## 🌟 Conclusion
This project demonstrates a creative use of OpenCV for real-time face detection and overlaying effects. The `FaceFilter` class is the core component, handling all aspects of image processing and visualization. The main function integrates these components to process video frames, apply the glasses overlay, and save the output video with consistent frame rates. This project can be extended for various applications such as augmented reality, video effects, and more.