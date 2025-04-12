import os
import cv2
import numpy as np
use_of_old = True
sift = cv2.xfeatures2d_SIFT.create()
train_folders = ['caltech-101_5_train/accordion',
                 'caltech-101_5_train/electric_guitar',
                 'caltech-101_5_train/grand_piano',
                 'caltech-101_5_train/mandolin',
                 'caltech-101_5_train/metronome']

def extract_local_features(path):
    img = cv2.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

# Creating the vocabulary
if os.path.isfile('vocabulary.npy') and use_of_old:
    print('Picking up the existing vocabulary')
    vocabulary = np.load('vocabulary.npy')
else:
    # Extract Database
    print('Extracting features...')
    train_descs = np.zeros((0, 128))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            desc = extract_local_features(path)
            if desc is None:
                continue
            train_descs = np.concatenate((train_descs, desc), axis=0)

    # Create vocabulary
    print('Creating vocabulary...')
    term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv2.BOWKMeansTrainer(50, term_crit, 1, cv2.KMEANS_PP_CENTERS)
    vocabulary = trainer.cluster(train_descs.astype(np.float32))

    np.save('vocabulary.npy', vocabulary)

if os.path.isfile('index.npy') and os.path.isfile('index.npy') and use_of_old:
    print('Picking up the existing indexing file')
    bow_descs = np.load('index.npy').astype(np.float32)
    print('Picking up the existing paths file')
    img_paths = np.load('paths.npy')
else:
    print('Creating index...')
    # Classification
    descriptor_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2SQR))
    descriptor_extractor.setVocabulary(vocabulary)

    img_paths = []
    bow_descs = np.zeros((0, vocabulary.shape[0]))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)

            img = cv2.imread(path)
            kp = sift.detect(img)
            bow_desc = descriptor_extractor.compute(img, kp)

            img_paths.append(path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

    np.save('index', bow_descs)
    np.save('paths', img_paths)
