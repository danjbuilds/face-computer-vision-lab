import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Locate the MediaPipe Face Landmarker model.
base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

# Configure the Face Landmarker.
#
# running_mode -> VIDEO enables tracking between frames.
# num_faces    -> Maximum number of faces to detect.
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

# Create the Face Landmarker.
detector = vision.FaceLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise Exception("cannot open cam")

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB images.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the NumPy image as a MediaPipe Image.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # VIDEO mode requires a timestamp for tracking.
    timestamp_ms = int(time.time() * 1000)

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.face_landmarks:

        face = result.face_landmarks[0]

        height, width, _ = frame.shape

        # Convert normalized coordinates (0.0-1.0)
        # into pixel coordinates for drawing.
        for landmark in face:

            pixel_x = int(landmark.x * width)
            pixel_y = int(landmark.y * height)

            cv2.circle(
                frame,
                (pixel_x, pixel_y),
                1,
                (0, 255, 0),
                -1
            )

    cv2.imshow("MediaPipe Face Landmarker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# MediaPipe Face Landmarker detects 478 facial landmarks.
#
# Pipeline:
#
# Camera Frame
#      ↓
# BGR → RGB
#      ↓
# MediaPipe Image
#      ↓
# Face Landmarker
#      ↓
# 478 Normalized Landmarks
#      ↓
# Pixel Coordinates (for drawing)