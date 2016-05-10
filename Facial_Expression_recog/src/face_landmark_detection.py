import dlib
import numpy as np
import features_extract
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        print "TooManyFaces"
        return None
    if len(rects) == 0:
        print "NoFaces"
        return None

    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        points = get_landmarks(frame)
        try:
            for point in points:
                frame = cv2.circle(frame, tuple(point), 2, (0,0,255), -1)
            cv2.imshow('frame',frame)
        except TypeError:
            continue
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

