import cv2
import mouthdetection
import smile_detection
import numpy as np
haar_face = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
haar_mouth = cv2.CascadeClassifier('../data/haarcascade_mouth.xml')

# video encoded in XVID
# TODO : try with other encoding
video_jordan = '../data/output.avi'

# kern = cv2.getGaborKernel((9, 9), 1.6, 0, np.pi/4, 1, 0, ktype=cv2.CV_32F)
#
#
# cap = cv2.VideoCapture(video_jordan)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if frame is not None:
#         detected_mouth = mouthdetection.findmouth(frame, haar_face, haar_mouth)[1]
#         frame = smile_detection.crop(detected_mouth, frame)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.filter2D(gray, cv2.CV_8UC3, kern)
#         print gray.shape
#         cv2.imshow('frame', gray)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()






# Parameters of getGaborKernel:
#
# ksize  Size of the filter returned.
# sigma  Standard deviation of the gaussian envelope.
# theta  Orientation of the normal to the parallel stripes of a Gabor function.
# lambd  Wavelength of the sinusoidal factor.
# gamma  Spatial aspect ratio.
# psi  Phase offset.
# ktype  Type of filter coefficients. It can be CV_32F or CV_64F .

thets = np.arange(0, np.pi, np.pi/8)


for thet in thets :
    kern = cv2.getGaborKernel(ksize = (9, 9), sigma = 3, theta = thet, lambd = 3.13,
                              gamma = 3, psi = 0, ktype=cv2.CV_32F)
    cap = cv2.VideoCapture(video_jordan)
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is not None:
            detected_object = mouthdetection.findmouth(frame, haar_face, haar_mouth)[0]
            frame = smile_detection.crop(detected_object, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.filter2D(gray, cv2.CV_8UC3, kern)
            edges = cv2.Canny(gray,100,100)
            cv2.imshow('frame', edges)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
