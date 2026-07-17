import cv2

# cv2.cvtColor()
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Color", frame)
    cv2.imshow("Gray", gray)

    # print(frame.shape)
    print(gray.shape)

    # print(frame[100,100])
    print(gray[100,100])

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()