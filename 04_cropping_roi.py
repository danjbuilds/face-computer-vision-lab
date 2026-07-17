import cv2 

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    cropped = frame[100:200, 200:400] #.copy()
    # cropped = frame[280:480, 440:640]
    # cropped = frame[0:200, 0:200]

    # cropped[:] = 0

    cv2.imshow("camera", frame)
    cv2.imshow("cropped", cropped)


    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()