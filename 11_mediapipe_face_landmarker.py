import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python # This imports MediaPipe's modern core setup tools.
from mediapipe.tasks.python import vision #  This imports the specific computer vision module. Because MediaPipe can also process audio and text, you must explicitly import vision to get access to visual AI models

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

    # result = detector.detect(mp_image)
    # print(result)

    timestamp_ms = int(time.time() * 1000)

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )
    # print(type(result))
    # print(result)

    if result.face_landmarks:

        # face = result.face_landmarks[0]
        # print(len(face))
        # landmark = face[0]

        # print(landmark.x)
        # print(landmark.y)
        # print(landmark.z)
    
        # height, width, _ = frame.shape

        # pixel_x = int(landmark.x * width)
        # pixel_y = int(landmark.y * height)

        # cv2.circle(
        #     frame,
        #     (pixel_x, pixel_y),
        #     1,
        #     (0, 255, 0),
        #     -1
        # )


        face = result.face_landmarks[0]

        height, width, _ = frame.shape

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