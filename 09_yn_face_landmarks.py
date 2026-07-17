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

        # Process every detected face.
        for face in faces:

            # Face bounding box.
            x, y, w, h = face[:4].astype(int)

            # Extract YuNet's five facial landmarks.
            # Each landmark is stored as an (x, y) pixel coordinate.
            right_eye = tuple(face[4:6].astype(int))
            left_eye = tuple(face[6:8].astype(int))
            nose = tuple(face[8:10].astype(int))
            right_mouth = tuple(face[10:12].astype(int))
            left_mouth = tuple(face[12:14].astype(int))

            # Store all landmarks in a list so they can be
            # processed together instead of individually.
            landmarks = [
                right_eye,
                left_eye,
                nose,
                right_mouth,
                left_mouth
            ]

            # Draw every landmark.
            #
            # Parameters:
            # point           -> (x, y) pixel coordinate
            # 3               -> Circle radius
            # (0,255,255)     -> Yellow (BGR)
            # -1              -> Filled circle
            for point in landmarks:
                cv2.circle(frame, point, 3, (0, 255, 255), -1)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Detection confidence score.
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

# Facial landmarks are predefined key points on a face.
#
# YuNet provides five landmarks:
# - Right eye
# - Left eye
# - Nose
# - Right mouth corner
# - Left mouth corner
#
# These landmarks are commonly used for:
# - Face alignment
# - Measuring facial geometry
# - Head pose estimation
# - Face tracking
#
# Unlike MediaPipe (478 landmarks), YuNet returns only
# five key landmarks, making it lightweight and fast.