import cv2
import mediapipe as mp
import time
import math

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

def to_pixel(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return (x, y)

blink_count = 0

eye_closed = False
closed_start_time = None
closed_duration = 0

eye_status = "Eyes Open"

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

        LEFT_EYE = [33, 160, 158, 133, 153, 144]

        height, width, _ = frame.shape

        # Get the six eye landmarks
        p1 = face[33]
        p2 = face[160]
        p3 = face[158]
        p4 = face[133]
        p5 = face[153]
        p6 = face[144]

        # Convert normalized coordinates to pixel coordinates
        pt1 = to_pixel(p1, width, height)
        pt2 = to_pixel(p2, width, height)
        pt3 = to_pixel(p3, width, height)
        pt4 = to_pixel(p4, width, height)
        pt5 = to_pixel(p5, width, height)
        pt6 = to_pixel(p6, width, height)

        # Draw the six landmarks
        for point in [pt1, pt2, pt3, pt4, pt5, pt6]:
            cv2.circle(frame, point, 4, (0, 255, 0), -1)

        # Draw the three measurements
        cv2.line(frame, pt2, pt6, (255, 0, 0), 2)
        cv2.line(frame, pt3, pt5, (255, 0, 0), 2)
        cv2.line(frame, pt1, pt4, (0, 0, 255), 2)

        # Calculate distances
        vertical1 = math.dist(pt2, pt6)
        vertical2 = math.dist(pt3, pt5)
        horizontal = math.dist(pt1, pt4)

        # Eye Aspect Ratio
        ear = (vertical1 + vertical2) / (2 * horizontal)
        # print(ear)

        THRESHOLD = 0.18

        if ear < THRESHOLD:

            if not eye_closed:
                eye_closed = True
                closed_start_time = time.time()

            closed_duration = time.time() - closed_start_time

            eye_status = "Eyes Closed"

        else:

            if eye_closed:

                closed_duration = time.time() - closed_start_time

                if closed_duration <= 0.30:
                    blink_count += 1

                eye_closed = False
                closed_start_time = None
                closed_duration = 0

            eye_status = "Eyes Open"

        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Closed: {closed_duration:.2f}s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"State: {eye_status}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)



    cv2.imshow("MediaPipe Face Landmarker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()