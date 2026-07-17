import cv2

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    # Crop a Region of Interest (ROI).
    #
    # Format:
    # frame[y_start:y_end, x_start:x_end]
    #
    # The first slice selects the rows (Y axis).
    # The second slice selects the columns (X axis).
    #
    # This extracts rows 100-199 and columns 200-399.
    cropped = frame[100:200, 200:400]

    cv2.imshow("camera", frame)
    cv2.imshow("cropped", cropped)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# ROI Examples
# -------------------------------------------------------------------

# Bottom-right region
# cropped = frame[280:480, 440:640]

# Top-left region
# cropped = frame[0:200, 0:200]

# -------------------------------------------------------------------
# ROI is a View (Not a Copy)
# -------------------------------------------------------------------

# This modifies every pixel inside the ROI.
# Since 'cropped' is a view of 'frame', the same region
# in the original image also becomes black.
#
# Use '.copy()' when creating the ROI if you want to edit
# the cropped image without changing the original frame.
#
# cropped[:] = 0