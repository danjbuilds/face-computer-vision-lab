import cv2

camera = cv2.VideoCapture(0)

# Create a YuNet face detector.
detector = cv2.FaceDetectorYN.create(
    model="models/face_detection_yunet_2026may.onnx",
    config="",
    input_size=(320,320)
)

while True:

    success, frame = camera.read()

    if not success:
        break

    # Mirror the webcam for a more natural view.
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # Match the detector input size to the current frame.
    detector.setInputSize((width, height))

    retval, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # Face bounding box.
            x, y, w, h = face[:4].astype(int)

            # Extract YuNet's five facial landmarks.
            right_eye = tuple(face[4:6].astype(int))
            left_eye = tuple(face[6:8].astype(int))
            nose = tuple(face[8:10].astype(int))
            right_mouth = tuple(face[10:12].astype(int))
            left_mouth = tuple(face[12:14].astype(int))

            landmarks = [
                right_eye,
                left_eye,
                nose,
                right_mouth,
                left_mouth
            ]

            # Draw each landmark.
            for point in landmarks:
                cv2.circle(frame, point, 3, (0, 255, 255), -1)

            # Draw facial geometry by connecting landmarks.
            #
            # Parameters:
            # image        -> Image to draw on
            # pt1, pt2     -> Start and end points
            # color        -> Line color (BGR)
            # thickness    -> Line width in pixels
            #
            # These connections help visualize facial orientation,
            # but they are NOT true head pose estimation.
            cv2.line(frame, right_eye, left_eye, (255, 255, 0), 2)
            cv2.line(frame, nose, right_eye, (0,255,0),2)
            cv2.line(frame, nose, left_eye, (0,255,0),2)

            cv2.rectangle(
                frame,
                (x,y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            score = face[-1]

            cv2.putText(
                frame,
                f"{score:.3f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YuNet_Face_Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# This experiment visualizes the spatial relationship
# between YuNet's five facial landmarks.
#
# Connecting landmarks makes it easier to observe
# how facial geometry changes as the head rotates.
#
# Camera Frame
#      ↓
# YuNet Face Detector
#      ↓
# 5 Facial Landmarks
#      ↓
# Landmark Connections
#
# True head pose estimation goes one step further by
# estimating yaw, pitch, and roll from facial landmarks
# using geometric methods (e.g., solvePnP) or AI models.