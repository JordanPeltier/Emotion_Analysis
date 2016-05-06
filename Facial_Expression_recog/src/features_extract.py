import numpy as np
import cv2
import smile_detection
import gauss_kernel
from scipy import ndimage
import time

haar_face = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

# video encoded in XVID
# TODO : try with other encoding
video_jordan = '../data/output.avi'


def get_dim(video):
    """
    return the 3 dimensions of a video : pixels in width and height and number of frames (time)
    :param video: path to a video on disk -> .avi container and Xvid enconding
    :return: a tuple with the 3 dimensions
    """
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(video)
    t = 0
    ret, frame = cap.read()

    if frame is not None:
        t += 1
        detectedface = haar_face.detectMultiScale(frame)

        # FACE: find the largest detected face as detected face
        maxfacesize = 0
        max_face = ()
        if detectedface.any():
            for face in detectedface:  # face: [0]: x; [1]: y; [2]: width; [3]: height
                if face[3] * face[2] > maxfacesize:
                    maxfacesize = face[3] * face[2]
                    max_face = face
    # noinspection PyUnboundLocalVariable
    frame = smile_detection.crop(max_face, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is not None:

            t += 1
            frame = smile_detection.crop(max_face, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return t, max_face[2], max_face[3]


def get_ndarray(video):
    """
    Tranform a sequence of frame into a nd array of the dimensions given by get_dim
    :param video: path to a video on disk -> .avi container and Xvid enconding
    :return: a ndarray
    """
    data = np.ndarray(shape=get_dim(video))
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    if frame is not None:
        t = 0
        detected_face = haar_face.detectMultiScale(frame)

        # FACE: find the largest detected face as detected face
        max_face_size = 0
        max_face = ()
        if detected_face.any():
            for face in detected_face:  # face: [0]: x; [1]: y; [2]: width; [3]: height
                if face[3] * face[2] > max_face_size:
                    max_face_size = face[3] * face[2]
                    max_face = face
    # noinspection PyUnboundLocalVariable
    frame = smile_detection.crop(max_face, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # noinspection PyUnboundLocalVariable
    data[t] = gray

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is not None:
            t += 1
            frame = smile_detection.crop(max_face, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data[t] = gray
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return data


def get_gauss_blur_video(video, dim, sig, tau):

    """
    Return the video in a ndarray format after convolution with a 3D gaussian kernel
    :param video: path to a video on disk -> .avi container and Xvid enconding
    :return: a ndarray format video after the convolution (blur in both space and time)
    """

    input_video = get_ndarray(video)
    input_video = input_video.astype(np.uint8)
    kern = gauss_kernel.gauss_kernel_3d(dim, sig, tau)
    gauss_kernel.plot_kern_3d(kern)


    print "\n" + "Beginning of linear scale space transformation (convolution with a 3D Gaussian kernel)"
    t0 = time.time()
    output = ndimage.convolve(input_video, kern)
    t1 = time.time()
    print "\n" + "End of linear scale space transformation (convolution with a 3D Gaussian kernel)"
    print "\n" + "Process time : " + str(t1-t0)


    output = output.astype(np.uint8)



    for frame in output:
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return output



foo = get_gauss_blur_video(video_jordan, 10, 10, 10)
print foo.shape


diff_t = np.diff(foo, n=1, axis=0)
diff_y = np.diff(foo, n=1, axis=1)
diff_x = np.diff(foo, n=1, axis=2)


print diff_x.shape, diff_y.shape, diff_t.shape

sec_mom_matrix = np.ndarray(shape=(foo.shape[0]-1, foo.shape[1]-1, foo.shape[2]-1), dtype = np.ndarray)
for tt in range(foo.shape[0]-1):
    for yy in range(foo.shape[1]-1):
        for xx in range(foo.shape[2]-1):
            x = int(diff_x[tt,yy,xx])
            y = int(diff_y[tt,yy,xx])
            t = int(diff_t[tt,yy,xx])
            sec_mom_matrix[tt, yy, xx] = np.array([[x^2, x*y, x*t],
                                                [x*y, y^2, y*t],
                                                [x*t, y*t, t^2]])




for frame in sec_mom_matrix:
    framebis = np.ndarray(shape=frame.shape)
    for i in frame[0,0]:
        framebis.append(i)
    cv2.imshow('frame', framebis)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# for frame in foo :
#     diff_x[t] = np.diff(frame, n = 1, axis = 1)
#     diff_y[t] = np.diff(frame, n = 1, axis = 0)
#     t+=1





# fgbg = cv2.createBackgroundSubtractorMOG2()
# frame = frame[20:250, 45:170]

# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# dst = cv2.cornerHarris(blur, 2, 5, 0.04)

# result is dilated for marking the corners, not important
# dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
# frame[dst > 0.01 * dst.max()] = [0, 0, 255]
