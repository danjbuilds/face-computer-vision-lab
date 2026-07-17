import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    frame = cv2.flip(frame, 1)

    if not success:
        break

    # Convert the color image (BGR) into a single-channel grayscale image.
    # Grayscale simplifies image processing when color information is unnecessary.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray", gray)

    # Compare the dimensions of a grayscale image with a color image.
    print(gray.shape)

    # A grayscale pixel stores one intensity value (0-255)
    # instead of separate Blue, Green, and Red values.
    print(gray[100,100])

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()