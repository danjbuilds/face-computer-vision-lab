import cv2 

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break


    original = frame

    blur5 = cv2.GaussianBlur(frame, (5,5), 0)

    blur21 = cv2.GaussianBlur(frame, (21,21), 0)

    cv2.imshow("Original", original)
    cv2.imshow("Blur 5", blur5)
    cv2.imshow("Blur 21", blur21)


    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()