import os
import cv2 as cv
import numpy as np

train_folders = ['mammals/train/brown_bear', 'mammals/train/camel', 'mammals/train/dolphin', 'mammals/train/giraffe', 'mammals/train/squirrel']

sift = cv.xfeatures2d_SIFT.create()


def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc


# Extract Database
print('Extracting features...')
labels = []
train_descs = np.zeros((0, 128))
for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        desc = extract_local_features(path)
        if desc is None:
            continue
        train_descs = np.concatenate((train_descs, desc), axis=0)
        labels.append(train_folders.index(folder))

# Create vocabulary
print('Creating vocabulary...')
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
trainer = cv.BOWKMeansTrainer(50, term_crit, 1, cv.KMEANS_PP_CENTERS)
vocabulary = trainer.cluster(train_descs.astype(np.float32))

np.save('vocabulary.npy', vocabulary)

print('Creating index...')
# Classification
descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

img_paths = []
# train_descs = np.zeros((0, 128))
bow_descs = np.zeros((0, vocabulary.shape[0]))
for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)

        img = cv.imread(path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        img_paths.append(path)
        bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

np.save('index.npy', bow_descs)
np.save('paths', img_paths)

#k-NN train model
labels = np.array(labels, np.int32)
knn = cv.ml.KNearest_create()
knn.train(bow_descs, cv.ml.ROW_SAMPLE, labels)
print('k-NN trained')

#SVM train model
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
labels = np.array(labels, np.int32)

svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
svm.save('svm')
print('svm trained')
