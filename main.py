import cv2  # type: ignore

camera = cv2.VideoCapture(0)  # Connects to the default built-in web camera (0)

# print(camera.isOpened()) # verify connection

# # Capture one frame
# success, frame = camera.read() # Grabs and decodes the very first frame from the camera

# if success:
#     cv2.imshow("dan by dan", frame) # Opens a window titled "Camera" and displays the frame
#     cv2.waitKey(0)       # Opens a window titled "Camera" and displays the frame

# camera.release() # Closes the camera hardware connection so other apps can use it
# cv2.destroyAllWindows() # Close every OpenCV window that is currently open.


# What is a Frame?

success, frame = camera.read()

# print(type(frame))
# print(frame) # numpy array
print(frame.shape)

print(frame[0,0])

print(frame[100,100])

print(frame[200,200])

camera.release()


# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# crop = frame[100:300, 200:400]

# small = cv2.resize(frame, (320, 240))

# blur = cv2.GaussianBlur(frame, (5,5), 0)

# Notice that each step produces another image that you can feed into the next step.

# Where we are now

# You now understand how to:

# ✅ Open a camera
# ✅ Capture frames
# ✅ Display frames
# ✅ Understand a frame as a NumPy array
# ✅ Convert to grayscale
# ✅ Crop a region of interest (ROI)
# ✅ Resize an image
# ✅ Blur an image