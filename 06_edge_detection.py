import cv2

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    # Convert the image to grayscale since edge detection
    # only needs intensity changes, not color information.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise and small details.
    # This helps prevent false edges from being detected.
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Detect edges using the Canny Edge Detection algorithm.
    #
    # Parameters:
    # 100 -> Lower threshold (weak edges)
    # 200 -> Upper threshold (strong edges)
    #
    # Pixels with intensity changes above the upper threshold
    # are considered edges, while weaker edges are kept only
    # if they connect to strong ones.
    edges = cv2.Canny(blur, 100, 200)

    cv2.imshow("Gray", gray)
    cv2.imshow("Blur", blur)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# Typical preprocessing pipeline:
# Color Image
#      ↓
# Grayscale
#      ↓
# Gaussian Blur
#      ↓
# Canny Edge Detection
#
# Edge detection highlights object boundaries by finding
# sudden changes in pixel intensity.