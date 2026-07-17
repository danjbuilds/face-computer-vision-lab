import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    print(frame.shape)

    # Create smaller and larger versions of the same frame.
    # Resizing changes the image dimensions but preserves the content.
    small = cv2.resize(frame, (320, 240))
    large = cv2.resize(frame, (1280, 960))

    print(small.shape)
    print(large.shape)

    # Display the original frame alongside the resized copies
    # to compare their resolutions.
    cv2.imshow("camera", frame)
    cv2.imshow("small", small)
    cv2.imshow("enlarge", large)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()