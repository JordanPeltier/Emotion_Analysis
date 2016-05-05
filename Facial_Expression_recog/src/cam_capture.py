import numpy as np
import cv2
import sys
import smile_detection

haarFace = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG2()
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
out = cv2.VideoWriter('../data/output.avi',fourcc, 20.0, frameSize=(640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    # frame = fgbg.apply(frame)
    if ret==True:

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


