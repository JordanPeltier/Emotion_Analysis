import numpy as np
import cv2
import sys
import smile_detection
import gauss_kernel
import scipy

# idea : first analyse the dim of the video in entrance to work with a fixed size nd array after


haarFace = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

video = '../data/output.avi'

def get_dim(video):
    cap = cv2.VideoCapture(video)
    t = 0
    ret, frame = cap.read()

    if frame is not None:
        t += 1
        detectedFace = haarFace.detectMultiScale(frame)

        # FACE: find the largest detected face as detected face
        maxFaceSize = 0
        maxFace = ()
        if detectedFace.any():
            for face in detectedFace:  # face: [0]: x; [1]: y; [2]: width; [3]: height
                if face[3] * face[2] > maxFaceSize:
                    maxFaceSize = face[3] * face[2]
                    maxFace = face
    frame = smile_detection.crop(maxFace, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    while (cap.isOpened()):
        ret, frame = cap.read()

        if frame is not None:
            t+=1
            frame = smile_detection.crop(maxFace, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return t, maxFace[2], maxFace[3]

def get_ndarray(video):


    data = np.ndarray(shape = get_dim(video))
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    if frame is not None:
        t = 0
        detectedFace = haarFace.detectMultiScale(frame)

        # FACE: find the largest detected face as detected face
        maxFaceSize = 0
        maxFace = ()
        if detectedFace.any():
            for face in detectedFace:  # face: [0]: x; [1]: y; [2]: width; [3]: height
                if face[3] * face[2] > maxFaceSize:
                    maxFaceSize = face[3] * face[2]
                    maxFace = face
    frame = smile_detection.crop(maxFace, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data[t] = gray
    cv2.imshow('frame', gray)

    while (cap.isOpened()):
        ret, frame = cap.read()

        if frame is not None:
            t += 1
            frame = smile_detection.crop(maxFace, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data[t] = gray


            cv2.imshow('frame', gray)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return data



get_ndarray(video)
kern = gauss_kernel.gauss_kernel_3d(5, 1, 1)


# fgbg = cv2.createBackgroundSubtractorMOG2()
# frame = frame[20:250, 45:170]

# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# dst = cv2.cornerHarris(blur, 2, 5, 0.04)

# result is dilated for marking the corners, not important
# dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
# frame[dst > 0.01 * dst.max()] = [0, 0, 255]
