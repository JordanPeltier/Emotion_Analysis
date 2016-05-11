import dlib as dl
import numpy as np
import cv2

haar_face = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
detector = dl.get_frontal_face_detector()
predictor = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        detected_face = haar_face.detectMultiScale(frame, minSize=(100, 100), minNeighbors = 5)
        # FACE: find the largest detected face as detected face
        try:
            if detected_face.any():
                for face in detected_face:  # face: [0]: x; [1]: y; [2]: width; [3]: height
                    face = face.astype(int)
                    l = face[0]
                    t = face[1]
                    r = face[0] + face[2]
                    b = face[1] + face[3]

                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

                    area_face = dl.rectangle(l, t, r, b)

                    landmarks = predictor(frame, area_face)
                    points = np.array([[p.x, p.y] for p in landmarks.parts()])
                    for point in points:
                        frame = cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        except AttributeError:
            print "No face detected"
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    else:
        break

cap.release()
cv2.destroyAllWindows()
