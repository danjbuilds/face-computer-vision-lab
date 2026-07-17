import cv2

# Open the default camera.
#
# Parameter:
# 0 -> First connected camera.
#     (1, 2, ...) selects other cameras if available.
camera = cv2.VideoCapture(0)

while True:

    # Capture one frame from the camera.
    #
    # Returns:
    # success -> True if a frame was captured.
    # frame   -> Image represented as a NumPy array.
    success, frame = camera.read()

    if not success:
        break

    # Mirror the webcam so movement appears natural,
    # similar to looking into a mirror.
    #
    # Parameter:
    # 1 -> Flip horizontally.
    frame = cv2.flip(frame, 1)

    # Image dimensions:
    # (height, width, color channels)
    print(frame.shape)

    # A pixel in a color image contains three values:
    # Blue, Green, and Red (BGR).
    print(frame[0, 0])
    print(frame[100, 100])
    print(frame[200, 200])

    # Display the current frame in a window.
    #
    # Parameters:
    # "Camera" -> Window title.
    # frame    -> Image to display.
    #
    # Continuously displaying successive frames creates
    # the illusion of live video.
    cv2.imshow("Camera", frame)

    # Wait 1 millisecond for a key press.
    #
    # If the user presses 'q', exit the program.
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera so other applications can use it.
camera.release()

# Close all OpenCV windows.
cv2.destroyAllWindows()

# ----------------------------------------------------------
# Notes
# ----------------------------------------------------------
#
# A webcam is simply a sequence of still images (frames).
#
# Camera
#     ↓
# Read Frame
#     ↓
# Process Frame
#     ↓
# Display Frame
#     ↓
# Repeat...
#
# Repeating this loop many times per second creates
# the appearance of live video.