import cv2

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    # Mirror the webcam for a more natural view.
    frame = cv2.flip(frame, 1)

    # Preprocessing pipeline before contour detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 100, 200)

    # Detect contours from the edge image.
    #
    # Parameters:
    # edges                 -> Binary edge image.
    # cv2.RETR_EXTERNAL     -> Retrieve only the outermost contours.
    # cv2.CHAIN_APPROX_SIMPLE -> Compress contour points to save memory.
    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw all detected contours.
    #
    # Parameters:
    # frame        -> Image to draw on.
    # contours     -> List of detected contours.
    # -1           -> Draw every contour.
    # (0,255,0)    -> Green color (BGR).
    # 2            -> Line thickness in pixels.
    cv2.drawContours(frame, contours, -1, (0,255,0), 2)

    cv2.imshow("Contours", frame)

    # Number of contours currently detected.
    print(len(contours))

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# A contour is a sequence of connected points that outlines
# the boundary of a detected object or shape.
#
# Typical contour detection pipeline:
#
# Color Image
#      ↓
# Grayscale
#      ↓
# Gaussian Blur
#      ↓
# Canny Edge Detection
#      ↓
# Find Contours
#
# Contours are commonly used for:
# - Object detection
# - Shape analysis
# - Measuring area or perimeter
# - Finding object boundaries