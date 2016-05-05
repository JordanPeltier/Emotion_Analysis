import cv2
import numpy as np
import smile_detection

haarFace = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)
if cam.isOpened():  # try to get the first frame
    rval, frame = cam.read()
    # frame : np array
else:
    rval = False
cv2.imshow("preview", frame)

detectedFace = haarFace.detectMultiScale(frame, minSize=(100, 100))

# FACE: find the largest detected face as detected face
maxFaceSize = 0
maxFace = ()
if detectedFace.any():
    for face in detectedFace:  # face: [0]: x; [1]: y; [2]: width; [3]: height
        if face[3] * face[2] > maxFaceSize:
            maxFaceSize = face[3] * face[2]
            maxFace = face
frame = smile_detection.crop(maxFace, frame)

im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
im_array = np.array(im_gray) # convert to np array

im_gray = np.float32(im_gray)
dst = cv2.cornerHarris(im_gray,2,5,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
frame[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',frame)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()