import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(frame, contours, -1, (0,255,0), 2)

    cv2.imshow("Contours", frame)

    print(len(contours))
    first = contours[0]
    print(first)

    if cv2.waitKey(1) == ord('q'):
        break



camera.release()
cv2.destroyAllWindows()