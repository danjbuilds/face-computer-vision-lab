import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from face_mesh_connections import FACEMESH_TESSELATION

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

        height, width, _ = frame.shape

        # FACEMESH_TESSELATION contains pairs of landmark indices.
        # Each pair defines two landmarks that should be connected.
        for start, end in FACEMESH_TESSELATION:

            point1 = face[start]
            point2 = face[end]

            # Convert normalized coordinates into pixel coordinates.
            x1 = int(point1.x * width)
            y1 = int(point1.y * height)

            x2 = int(point2.x * width)
            y2 = int(point2.y * height)

            # Draw one edge of the facial mesh.
            #
            # Parameters:
            # (x1, y1) -> Starting landmark
            # (x2, y2) -> Ending landmark
            # (0,255,0)-> Green (BGR)
            # 1        -> Line thickness
            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1
            )

    cv2.imshow("MediaPipe Face Landmarker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# A face mesh is created by connecting facial landmarks.
#
# FACEMESH_TESSELATION is a predefined collection of
# landmark pairs supplied by MediaPipe.
#
# Pipeline:
#
# Face Landmarker
#       ↓
# 478 Facial Landmarks
#       ↓
# FACEMESH_TESSELATION
#       ↓
# Complete Face Mesh
#
# The face mesh represents the geometry of the face and
# serves as the foundation for applications such as
# facial expression analysis, blink detection, head pose,
# and face tracking.