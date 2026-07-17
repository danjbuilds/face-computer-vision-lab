import cv2

camera = cv2.VideoCapture(0)

# Create a YuNet face detector using a pretrained ONNX model.
#
# Parameters:
# model      -> Path to the trained YuNet model.
# config     -> Additional model configuration (unused for YuNet).
# input_size -> Expected image size for inference.
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

    # Get the current frame dimensions.
    height, width = frame.shape[:2]

    # Update the detector with the actual frame size.
    # YuNet expects the detector input size to match the image being processed.
    detector.setInputSize((width, height))

    # Run face detection.
    #
    # Returns:
    # retval -> Number of detected faces.
    # faces  -> NumPy array containing one row per detected face.
    retval, faces = detector.detect(frame)

    if faces is not None:

        # Process every detected face.
        for face in faces:

            # Bounding box:
            # x, y -> Top-left corner
            # w, h -> Width and height
            x, y, w, h = face[:4].astype(int)

            # YuNet predicts five facial landmarks:
            # right eye, left eye, nose, right mouth, left mouth
            right_eye = tuple(face[4:6].astype(int))
            left_eye = tuple(face[6:8].astype(int))
            nose = tuple(face[8:10].astype(int))
            right_mouth = tuple(face[10:12].astype(int))
            left_mouth = tuple(face[12:14].astype(int))

            cv2.circle(frame, right_eye, 3, (255, 0, 0), -1)
            cv2.circle(frame, left_eye, 3, (255, 0, 0), -1)
            cv2.circle(frame, nose, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_mouth, 3, (0, 0, 255), -1)
            cv2.circle(frame, left_mouth, 3, (0, 0, 255), -1)

            # Draw the detected face bounding box.
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Detection confidence score.
            # Higher values indicate greater confidence that a face was detected.
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

# YuNet is an AI-based face detector.
#
# Output for each detected face includes:
# - Bounding box (x, y, width, height)
# - Five facial landmarks
#   • Right eye
#   • Left eye
#   • Nose
#   • Right mouth corner
#   • Left mouth corner
# - Detection confidence score
#
# Typical pipeline:
#
# Camera Frame
#      ↓
# YuNet Face Detector
#      ↓
# Face Bounding Box + 5 Landmarks
#
# These landmarks are useful for face alignment,
# head pose estimation, and as a starting point
# for more advanced face analysis.