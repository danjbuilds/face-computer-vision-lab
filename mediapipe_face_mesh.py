import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python # This imports MediaPipe's modern core setup tools.
from mediapipe.tasks.python import vision #  This imports the specific computer vision module. Because MediaPipe can also process audio and text, you must explicitly import vision to get access to visual AI models
from face_mesh_connections import FACEMESH_TESSELATION


base_options = python.BaseOptions( #  This object is a basic file finder used by all MediaPipe tools.
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
    raise Exception("connot open cam")


while True:

    success, frame = camera.read()

    if not success:
        break
    
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image

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

        for start, end in FACEMESH_TESSELATION:

            point1 = face[start]
            point2 = face[end]

            x1 = int(point1.x * width)
            y1 = int(point1.y * height)

            x2 = int(point2.x * width)
            y2 = int(point2.y * height)

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