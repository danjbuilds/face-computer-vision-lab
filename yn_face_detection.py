import cv2 

camera = cv2.VideoCapture(0)

detector = cv2.FaceDetectorYN.create(
    model="models/face_detection_yunet_2026may.onnx",
    config="",
    input_size=(320,320)
)

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    retval, faces = detector.detect(frame)

    # print(retval)
    # print(faces)

    # if faces is not None:

    #     face = faces[0]

    #     x, y, w, h = face[:4]

    #     print(f"x = {x:.1f}")
    #     print(f"y = {y:.1f}")
    #     print(f"width = {w:.1f}")
    #     print(f"height = {h:.1f}")

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

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

            cv2.rectangle(
                frame,
                (x,y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # print(f"Confidence: {face[-1]:.3f}")
            
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


# ✅ Read frames from a camera.
# ✅ Basic image processing.
# ✅ Use a modern face detector.
# ✅ Understand that detections are NumPy arrays.
# ✅ Extract the bounding box.
# ✅ Extract facial landmarks.
# ✅ Use those landmarks for drawing.