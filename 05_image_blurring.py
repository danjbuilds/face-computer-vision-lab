import cv2

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    original = frame

    # Apply Gaussian Blur to smooth the image.
    #
    # Parameters:
    # (5,5)  -> Kernel size (width, height). Larger kernels produce stronger blur.
    # 0      -> Sigma (standard deviation). A value of 0 lets OpenCV calculate it automatically.
    blur5 = cv2.GaussianBlur(frame, (5,5), 0)

    # Larger kernel = stronger smoothing.
    # More image details and noise are removed.
    blur21 = cv2.GaussianBlur(frame, (21,21), 0)

    # Compare the original image with different blur strengths.
    cv2.imshow("Original", original)
    cv2.imshow("Blur 5", blur5)
    cv2.imshow("Blur 21", blur21)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

# Image blurring smooths neighboring pixels together.
# It is commonly used to:
# - Reduce image noise
# - Remove small details
# - Prepare images before edge detection or contour detection
#
# Increasing the kernel size increases the amount of blur.