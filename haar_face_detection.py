import cv2 

camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

while True:

    success, frame = camera.read()
    

    if not success:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x,y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


    cv2.imshow("Face_Detection", frame)
  

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()