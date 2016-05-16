import cv2
import dlib as dl
import os
import shelve
import cPickle


train_1 = "/home/jordan/Documents/data/HELEN/train_1"
train_2 = "/home/jordan/Documents/data/HELEN/train_2"
train_3 = "/home/jordan/Documents/data/HELEN/train_3"
train_4 = "/home/jordan/Documents/data/HELEN/train_4"
test = "/home/jordan/Documents/data/HELEN/test"


train = list()

for item in os.listdir(train_1):
    if '.jpg' in item:
        image = cv2.imread(os.path.join(train_1, item), 0)
        if image is not None:
            train.append(image)
            print "Done"

for item in os.listdir(train_2):
    if '.jpg' in item:
        image = cv2.imread(os.path.join(train_2, item), 0)
        if image is not None:
            train.append(image)
            print "Done"

for item in os.listdir(train_3):
    if '.jpg' in item:
        image = cv2.imread(os.path.join(train_3, item), 0)
        if image is not None:
            train.append(image)
            print "Done"

for item in os.listdir(train_4):
    if '.jpg' in item:
        image = cv2.imread(os.path.join(train_4, item), 0)
        if image is not None:
            train.append(image)
            print "Done"

detector = dl.get_frontal_face_detector()

for image in train:
     face_train.append(detector(image, 1))

f = open("~/PycharmProjects/Emotion_Analysis/Facial_Expression_recog/data/face_train.save", "rb")
face_train = cPickle(f)