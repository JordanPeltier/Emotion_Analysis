import numpy as np
import cv2
import smile_detection
import gauss_kernel
from scipy import ndimage
import time
import shelve

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
    y = frame.shape[0]
    x = frame.shape[1]
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
    return t, y, x


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
    cv2.normalize(gray, gray, 0, 255, norm_type=cv2.NORM_MINMAX)
    # noinspection PyUnboundLocalVariable
    data[t] = gray

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is not None:
            t += 1
            frame = smile_detection.crop(max_face, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.normalize(gray, gray, 0, 255, norm_type=cv2.NORM_MINMAX)
            data[t] = gray

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return data


def get_gauss_blur_video(video, dims, dimt, sig, tau):

    """
    Return the video in a ndarray format after convolution with a 3D gaussian kernel
    :param video: path to a video on disk -> .avi container and Xvid enconding
    :return: a ndarray format video after the convolution (blur in both space and time)
    """

    input_video = get_ndarray(video)
    input_video = input_video.astype(np.uint8)
    kern = gauss_kernel.gauss_kernel_3d(dims, dimt, sig, tau)
    # gauss_kernel.plot_kern_3d(kern)


    print "\n" + "Beginning of linear scale space transformation (convolution with a 3D Gaussian kernel)"
    t0 = time.time()
    output = ndimage.convolve(input_video, kern)
    t1 = time.time()
    print "\n" + "End of linear scale space transformation (convolution with a 3D Gaussian kernel)"
    print "\n" + "Process time : " + str(t1-t0)


    output = output.astype(np.uint8)
    cv2.normalize(output, output, 0, 255, norm_type=cv2.NORM_MINMAX)



    for frame in output:
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return output

def get_moment_matrix(video, dims, dimt, sig, tau):
    blur_video = get_gauss_blur_video(video, dims, dimt, sig, tau)

    diff_t = np.diff(blur_video, n=1, axis=0)
    diff_y = np.diff(blur_video, n=1, axis=1)
    diff_x = np.diff(blur_video, n=1, axis=2)

    moment_matrix = np.ndarray(shape=(blur_video.shape[0]-1, blur_video.shape[1]-1, blur_video.shape[2]-1), dtype = np.ndarray)

    for tt in range(blur_video.shape[0]-1):
        for yy in range(blur_video.shape[1]-1):
            for xx in range(blur_video.shape[2]-1):
                x = int(diff_x[tt,yy,xx])
                y = int(diff_y[tt,yy,xx])
                t = int(diff_t[tt,yy,xx])
                moment_matrix[tt, yy, xx] = np.array([[x^2, x*y, x*t],
                                                       [x*y, y^2, y*t],
                                                       [x*t, y*t, t^2]])
                moment_matrix[tt, yy, xx] = moment_matrix[tt, yy, xx].astype(np.uint8)


    return moment_matrix


def get_xx(moment_matrix):
    moment_matrix_xx = np.ndarray(shape=(moment_matrix.shape[0], moment_matrix.shape[1], moment_matrix.shape[2]), dtype=float)
    for tt in range(moment_matrix.shape[0]):
        for yy in range(moment_matrix.shape[1]):
            for xx in range(moment_matrix.shape[2]):
                moment_matrix_xx[tt, yy, xx] = moment_matrix[tt, yy, xx][0,0]


    return moment_matrix_xx
def get_xy(moment_matrix):
    moment_matrix_xy = np.ndarray(shape=(moment_matrix.shape[0], moment_matrix.shape[1], moment_matrix.shape[2]), dtype = float)
    for tt in range(moment_matrix.shape[0]):
        for yy in range(moment_matrix.shape[1]):
            for xx in range(moment_matrix.shape[2]):
                moment_matrix_xy[tt, yy, xx] = moment_matrix[tt, yy, xx][1,0]

    return moment_matrix_xy
def get_xt(moment_matrix):
    moment_matrix_xt = np.ndarray(shape=(moment_matrix.shape[0], moment_matrix.shape[1], moment_matrix.shape[2]), dtype = float)
    for tt in range(moment_matrix.shape[0]):
        for yy in range(moment_matrix.shape[1]):
            for xx in range(moment_matrix.shape[2]):
                moment_matrix_xt[tt, yy, xx] = moment_matrix[tt, yy, xx][0,2]

    return moment_matrix_xt
def get_yt(moment_matrix):
    moment_matrix_yt = np.ndarray(shape=(moment_matrix.shape[0], moment_matrix.shape[1], moment_matrix.shape[2]), dtype = float)
    for tt in range(moment_matrix.shape[0]):
        for yy in range(moment_matrix.shape[1]):
            for xx in range(moment_matrix.shape[2]):
                moment_matrix_yt[tt, yy, xx] = moment_matrix[tt, yy, xx][1,2]

    return moment_matrix_yt
def get_yy(moment_matrix):
    moment_matrix_yy = np.ndarray(shape=(moment_matrix.shape[0], moment_matrix.shape[1], moment_matrix.shape[2]), dtype = float)
    for tt in range(moment_matrix.shape[0]):
        for yy in range(moment_matrix.shape[1]):
            for xx in range(moment_matrix.shape[2]):
                moment_matrix_yy[tt, yy, xx] = moment_matrix[tt, yy, xx][1,1]

    return moment_matrix_yy
def get_tt(moment_matrix):
    moment_matrix_tt = np.ndarray(shape=(moment_matrix.shape[0], moment_matrix.shape[1], moment_matrix.shape[2]), dtype = float)
    for tt in range(moment_matrix.shape[0]):
        for yy in range(moment_matrix.shape[1]):
            for xx in range(moment_matrix.shape[2]):
                moment_matrix_tt[tt, yy, xx] = moment_matrix[tt, yy, xx][2,2]

    return moment_matrix_tt


def display_moment_matrix(moment_matrix, direction="tt"):

    options = {"xx" : get_xx(moment_matrix),
               "xy" : get_xy(moment_matrix),
               "xt" : get_xt(moment_matrix),
               "yt" : get_yt(moment_matrix),
               "yy" : get_yy(moment_matrix),
               "tt" : get_tt(moment_matrix)
              }

    moment_matrix_dir = options[direction]

    for frame in moment_matrix_dir:
        cv2.imshow('frame', frame)
        if cv2.waitKey(250) & 0xFF == ord('q'):
            break

def get_h(video, dims = 5, dimt = 5, sig = np.sqrt(2), tau = np.sqrt(2), s = 1):
    matrix = get_moment_matrix(video, dims, dimt, sig, tau)
    xx = get_xx(matrix)
    cv2.normalize(xx, xx, 0, 255, norm_type=cv2.NORM_MINMAX)

    xy = get_xy(matrix)
    cv2.normalize(xy, xy, 0, 255, norm_type=cv2.NORM_MINMAX)

    xt = get_xt(matrix)
    cv2.normalize(xt, xt, 0, 255, norm_type=cv2.NORM_MINMAX)

    yt = get_yt(matrix)
    cv2.normalize(yt, yt, 0, 255, norm_type=cv2.NORM_MINMAX)

    yy = get_yy(matrix)
    cv2.normalize(yy, yy, 0, 255, norm_type=cv2.NORM_MINMAX)

    tt = get_tt(matrix)
    cv2.normalize(tt, tt, 0, 255, norm_type=cv2.NORM_MINMAX)


    kern = gauss_kernel.gauss_kernel_3d(dims, dimt, s*sig, s*tau)

    mu_xx = ndimage.convolve(xx, kern)
    cv2.normalize(mu_xx, mu_xx, 0, 255, norm_type=cv2.NORM_MINMAX)

    mu_xy = ndimage.convolve(xy, kern)
    cv2.normalize(mu_xy, mu_xy, 0, 255, norm_type=cv2.NORM_MINMAX)

    mu_xt = ndimage.convolve(xt, kern)
    cv2.normalize(mu_xt, mu_xt, 0, 255, norm_type=cv2.NORM_MINMAX)

    mu_yt = ndimage.convolve(yt, kern)
    cv2.normalize(mu_yt, mu_yt, 0, 255, norm_type=cv2.NORM_MINMAX)

    mu_yy = ndimage.convolve(yy, kern)
    cv2.normalize(mu_yy, mu_yy, 0, 255, norm_type=cv2.NORM_MINMAX)

    mu_tt = ndimage.convolve(tt, kern)
    cv2.normalize(mu_tt, mu_tt, 0, 255, norm_type=cv2.NORM_MINMAX)

    H = np.ndarray(shape = matrix.shape, dtype = float)

    for t in range(H.shape[0]):
        for y in range(H.shape[1]):
            for x in range(H.shape[2]):
                mu = np.matrix([[mu_xx[t, y, x], mu_xy[t, y, x], mu_xt[t, y, x]],
                                [mu_xy[t, y, x], mu_yy[t, y, x], mu_yt[t, y, x]],
                                [mu_xt[t, y, x], mu_yt[t, y, x], mu_tt[t, y, x]]])
                H[t, y, x] = np.linalg.det(mu) - 0.06*np.trace(mu)**3

    for frame in H:
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    return H



# h = get_h(video_jordan)

# my_shelf = shelve.open("shelve.out")
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()
#
# for frame in h:
#     cv2.imshow("frame", frame)
#     print frame
#     cv2.waitKey(500)

###################################################################
# filename='shelve.out'
# my_shelf = shelve.open(filename,'n') # 'n' for new
#
# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         #
#         # __builtins__, my_shelf, and imported modules can not be shelved.
#         #
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()
###################################################################




# fgbg = cv2.createBackgroundSubtractorMOG2()
# frame = frame[20:250, 45:170]

# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# dst = cv2.cornerHarris(blur, 2, 5, 0.04)

# result is dilated for marking the corners, not important
# dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
# frame[dst > 0.01 * dst.max()] = [0, 0, 255]
