# Computer Vision Learning Repository

This repository documents my hands-on journey learning computer vision using **OpenCV**, **YuNet**, and **MediaPipe** with Python.

The files are intentionally numbered in the order I learned each concept. Each lab introduces one new idea while building upon previous lessons.

---

## Requirements

```bash
pip install opencv-python
pip install mediapipe
```

---

## Repository Structure

### OpenCV Fundamentals

| File | Topic | Description |
|------|-------|-------------|
| 01_live_video.py | Live Video | Capture frames from a webcam, inspect pixel values, and display a live video stream. |
| 02_color_spaces_grayscale.py | Grayscale | Convert color images from BGR to grayscale and understand color channels. |
| 03_resizing_images.py | Image Resizing | Resize images to different resolutions. |
| 04_cropping_roi.py | Region of Interest (ROI) | Crop portions of an image using NumPy slicing. |
| 05_image_blurring.py | Gaussian Blur | Smooth images to reduce noise before further processing. |
| 06_edge_detection.py | Canny Edge Detection | Detect image edges using the Canny algorithm. |
| 07_contours.py | Contours | Extract object boundaries from edge images. |

---

### YuNet Face Detection

| File | Topic | Description |
|------|-------|-------------|
| 08_yn_face_detection.py | Face Detection | Detect faces using YuNet and obtain bounding boxes and confidence scores. |
| 09_yn_face_landmarks.py | Facial Landmarks | Extract and visualize YuNet's five facial landmarks. |
| 10_yn_head_pose.py | Facial Geometry | Connect facial landmarks to visualize facial orientation and geometry. *(Visualization only; not true head pose estimation.)* |

---

### MediaPipe Face Landmarker

| File | Topic | Description |
|------|-------|-------------|
| 11_mediapipe_face_landmarker.py | Face Landmarker | Detect and display MediaPipe's 478 facial landmarks. |
| 12_mediapipe_face_mesh.py | Face Mesh | Draw the complete facial mesh using MediaPipe's predefined landmark connections. |
| face_mesh_connections.py | Helper Module | Contains MediaPipe's face mesh connection definitions used by the mesh visualization examples. |

---

### Blink Detection

| File | Topic | Description |
|------|-------|-------------|
| 13_blink_detection_lab.py | Blink Detection Lab | Proof-of-concept experiment measuring eyelid distance using two landmarks. |
| 14_eye_blink_detection.py | Eye Aspect Ratio (EAR) | Implement blink detection using the Eye Aspect Ratio algorithm with six eye landmarks. |

---

## Learning Progression

```
OpenCV Fundamentals
        ↓
Face Detection (YuNet)
        ↓
Facial Landmarks
        ↓
Face Mesh
        ↓
Blink Detection
        ↓
Eye Aspect Ratio (EAR)
```

Each lab focuses on a single concept so that later examples remain easy to understand and revisit.