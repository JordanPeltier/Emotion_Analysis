import csv
import numpy as np
import time
from PIL import Image
import cv2
import mouthdetection as m
import logistic

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector
haarFace = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
haarMouth = cv2.CascadeClassifier('../data/haarcascade_mouth.xml')

"""
pop up an image showing the mouth with a blue rectangle
"""
def show(area, img):
    cv2.Rectangle(img,(area[0],area[1]),
                     (area[0]+area[2],area[1]+area[3]),
                    (255,0,0),2)
    cv2.NamedWindow('Face Detection', cv2.CV_WINDOW_NORMAL)
    cv2.ShowImage('Face Detection', img)
    cv2.WaitKey()

"""
given an area to be cropped, crop() returns a cropped image
"""
def crop(area, img):
    crop = img[area[1]:area[1] + area[3]+30, area[0]:area[0]+area[2]] #img[y: y + h, x: x + w]
    return crop

"""
given a ndarray image, vectorize the grayscale pixels to
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""
def vectorize_ndarray(frame):
    size = WIDTH, HEIGHT # (width, height)
    resized_im = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)# resize image
    im_grey = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array

"""
given a jpg image, vectorize the grayscale pixels to
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""
def vectorize_jpeg(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = Image.open(filename)
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array

if __name__ == '__main__':
    """
    load training data
    """
    # create a list for filenames of smiles pictures
    smilefiles = []
    with open('../data/smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open('../data/neutral.csv', 'rb') as csvfile: # rb mode is for read as a binary file
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load smile data
    PATH = "../data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize_jpeg(PATH + filename)
        labels.append(1)

    # load neutral data    
    PATH = "../data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize_jpeg(PATH + filename)
        labels.append(0)

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    

    """
    open webcam and capture images
    """
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        # frame : np array
    else:
        rval = False

    print "\n\n\n\n\npress q to exit"


    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        if cv2.waitKey(10) & 0xFF == ord('q'): # exit on ESC
            break
        else :
            t0 = time.time()
            face, mouth = m.findmouth(frame, haarFace, haarMouth)
            t1 = time.time()
            print t1 - t0


            try:
                (x, y, w, h) = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                (x, y, w, h) = mouth
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            except ValueError:
                continue

            # shows(mouth)
            if len(mouth) != 0: # did not return error
                mouthimg = crop(mouth, frame)

                # predict the captured emotion

                result = lr.predict(vectorize_ndarray(mouthimg))


                if result == 1:
                    print "you are smiling! :-) "
                else:
                    print "you are not smiling :-| "



            else:
                print "failed to detect mouth. Try hold your head straight and make sure there is only one face."
    
    cv2.destroyWindow("preview")
