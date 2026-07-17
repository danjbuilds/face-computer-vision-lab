import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise Exception("cannot open cam")

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(time.time() * 1000)

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.face_landmarks:

        face = result.face_landmarks[0]

        # Select two eyelid landmarks.
        # Landmark 159 -> Upper eyelid
        # Landmark 145 -> Lower eyelid
        upper_eyelid = face[159]
        lower_eyelid = face[145]

        height, width, _ = frame.shape

        # Convert normalized coordinates into pixel coordinates.
        x1 = int(upper_eyelid.x * width)
        y1 = int(upper_eyelid.y * height)

        x2 = int(lower_eyelid.x * width)
        y2 = int(lower_eyelid.y * height)

        cv2.circle(frame, (x1, y1), 4, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 4, (0, 255, 0), -1)

        # Connect the eyelid landmarks for visualization.
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255,0,0),
            1
        )

        # Measure the distance between the eyelids.
        # Smaller distance -> Eye is closing.
        # Larger distance -> Eye is open.
        distance = math.dist(
            (x1, y1),
            (x2, y2)
        )

        cv2.putText(
            frame,
            f"{distance:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Eye Points", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# This lab demonstrates the simplest approach to estimating
# eye openness using only two eyelid landmarks.
#
# Pipeline:
#
# Face Landmarker
#       ↓
# Two Eyelid Landmarks
#       ↓
# Pixel Coordinates
#       ↓
# Distance Between Eyelids
#
# The measured distance decreases as the eye closes.
#
# This is only a proof of concept. The next lab introduces
# Eye Aspect Ratio (EAR), which uses six landmarks and
# provides a more robust blink detection algorithm.