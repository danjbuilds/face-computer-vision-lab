import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Locate the MediaPipe Face Landmarker model.
base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

# Configure the Face Landmarker.
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


# Convert a normalized MediaPipe landmark into pixel coordinates.
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

        # Six landmarks that define the left eye.
        #
        # p2 ----- p3
        # |         |
        # p1       p4
        # |         |
        # p6 ----- p5
        p1 = face[33]
        p2 = face[160]
        p3 = face[158]
        p4 = face[133]
        p5 = face[153]
        p6 = face[144]

        # Convert normalized coordinates into pixel coordinates.
        pt1 = to_pixel(p1, width, height)
        pt2 = to_pixel(p2, width, height)
        pt3 = to_pixel(p3, width, height)
        pt4 = to_pixel(p4, width, height)
        pt5 = to_pixel(p5, width, height)
        pt6 = to_pixel(p6, width, height)

        # Uncomment to visualize the EAR measurements.
        #
        # for point in [pt1, pt2, pt3, pt4, pt5, pt6]:
        #     cv2.circle(frame, point, 4, (0, 255, 0), -1)
        #
        # cv2.line(frame, pt2, pt6, (255, 0, 0), 2)
        # cv2.line(frame, pt3, pt5, (255, 0, 0), 2)
        # cv2.line(frame, pt1, pt4, (0, 0, 255), 2)

        # Measure the eye.
        #
        # Vertical distances:
        #   p2 ↔ p6
        #   p3 ↔ p5
        #
        # Horizontal distance:
        #   p1 ↔ p4
        vertical1 = math.dist(pt2, pt6)
        vertical2 = math.dist(pt3, pt5)
        horizontal = math.dist(pt1, pt4)

        # Eye Aspect Ratio (EAR)
        #
        # EAR = (vertical1 + vertical2) / (2 × horizontal)
        #
        # As the eye closes:
        # - Vertical distances decrease.
        # - Horizontal distance stays nearly constant.
        # - EAR becomes smaller.
        ear = (vertical1 + vertical2) / (2 * horizontal)

        # EAR threshold for deciding whether the eye is closed.
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

                # Count a blink only if the eye was closed briefly.
                if closed_duration <= 0.30:
                    blink_count += 1

                eye_closed = False
                closed_start_time = None
                closed_duration = 0

            eye_status = "Eyes Open"

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Closed: {closed_duration:.2f}s",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"State: {eye_status}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2)

    cv2.imshow("Blink Detection (EAR)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# Eye Aspect Ratio (EAR) estimates eye openness using six landmarks.
#
#           p2 -------- p3
#          /            \
#        p1              p4
#          \            /
#           p6 -------- p5
#
# EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
#
# Pipeline:
#
# Face Landmarker
#       ↓
# Six Eye Landmarks
#       ↓
# Eye Aspect Ratio (EAR)
#       ↓
# Eye Open / Eye Closed
#       ↓
# Blink Counter
#
# Unlike measuring a single eyelid distance, EAR is normalized by
# the eye's width, making it much more robust to changes in face
# size and distance from the camera.
#
# A blink is detected when:
# 1. EAR falls below the threshold.
# 2. The eye remains closed briefly.
# 3. The eye opens again.