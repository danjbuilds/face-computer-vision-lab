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

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x,y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

    cv2.imshow("YuNet_Face_Detection", frame)
  

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()