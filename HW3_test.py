import os
import cv2 as cv
import numpy as np


sift = cv.xfeatures2d_SIFT.create()

#Loading the neccessary
vocabulary = np.load('vocabulary.npy')
img_paths = np.load('paths.npy')
bow_descs = np.load('index.npy').astype(np.float32)

descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)


#Loading and reading a random image
test_img = 'mammals/test/brown_bear/brown_bear-0139.jpg'
img = cv.imread(test_img)
cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test', img)
cv.waitKey()
kp = sift.detect(img)
bow_desc = descriptor_extractor.compute(img, kp)


print('k-NN testing shows that...')

knn = cv.ml.KNearest_create()
response, results, neighbours, dist = knn.findNearest(bow_desc, 3)

if response ==1:
    print("It's a brown bear")
elif response == 2:
    print("It's a brown bear")
elif response == 3:
    print("It's a dolphin")
elif response == 4:
    print("It's a giraffe")
elif response == 5:
    print("It's a squirrel")

print('svm testing shows that... ')
# Load SVM
svm = cv.ml.SVM_create()
svm = svm.load('svm')
response = svm.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)

if response[1] < 0:
    print('It is a brown_bear')
else:
    print('It ')
