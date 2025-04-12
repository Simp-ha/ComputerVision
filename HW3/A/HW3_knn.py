import os
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
from BOVW import vocabulary, bow_descs, img_paths
tots = 0
if sys.argv == True: 
    plot = sys.argv[1] 
else:
    plot = True

test_folders = ['caltech-101_5_test/accordion',
                'caltech-101_5_test/electric_guitar',
                'caltech-101_5_test/grand_piano',
                'caltech-101_5_test/mandolin',
                'caltech-101_5_test/metronome']
classes = np.array([x[19:] for x in test_folders])

# bow_descs = np.load('index.npy').astype(np.float32)
#
# img_paths = np.load('paths.npy')

# TRAINING kNN
labels = []
for p in img_paths:
    if classes[0] in p:
        labels.append(0)
    elif classes[1] in p:
        labels.append(1)
    elif classes[2] in p:
        labels.append(2)
    elif classes[3] in p:
        labels.append(3)
    else:
        labels.append(4)

labels = np.array(labels, np.int32)

knn = cv2.ml.KNearest_create()
knn.train(bow_descs, cv2.ml.ROW_SAMPLE, labels)


# TESTING kNN
sift = cv2.xfeatures2d_SIFT.create()
descriptor_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

# List of successes of every neighbour value in every class (aka a 17x5 list)
success = []
# Different neighbours per class
for k in range(3, 20):
    # print(f'\n=== {k} neighbours ===')
    # List of successes of every class
    class_success = []
    tot = []
    for i, folder in enumerate(test_folders):
        files = os.listdir(folder)
        # Success counter per class (for every image)
        S_counter_class = 0
        for test_img in files:
            path = os.path.join(folder, test_img)
            img = cv2.imread(path)
            kp = sift.detect(img)
            bow_desc = descriptor_extractor.compute(img, kp)
            response, results, neighbours, dist = knn.findNearest(bow_desc, k)
            # Correct predictions
            if response == test_folders.index(folder):
                # print(f'It is actually a {folder[19:]}')
                S_counter_class += 1
            # print(f'Prediction is a {test_folders[int(response)][19:]}')
        tot.append(len(files))
        # Success Rate of every class
        # print(f'Successes of {classes[i]} = {S_counter_class}/{len(files)}')
        class_success.append(S_counter_class)
    # Success rate of all
    # print(f'Success rate = {round(sum(class_success)/sum(tot)*100, 2)}%')
    success.append(class_success)

# suc_rate  = []
# neighbours = len(range(3,20))
# for i in range (0,neighbours):

if plot:
    x = np.arange(len(classes))
    # No. neighbours
    neighbours = len(range(3,20))
    # Width of bars
    width = 0.8/neighbours
    fig, ax = plt.subplots(figsize=(12,8))
    colors = plt.cm.get_cmap('tab20', neighbours)
    bars = []
    # X: Classes, Y: Success Rate, COLORS: neighbours
    for i in range (0,neighbours):
        # Calculation of the success rate of number of neighbours
        suc_rate  = [a/b for a,b in zip(success[i],tot)]
        (ax.bar(x + i * width - (neighbours / 2) * width, suc_rate, width, label=f'Neighbours {i+3}', color=colors(i)))

    ax.set_xlabel('Classes')
    ax.set_ylabel('Success Rate')
    ax.set_title('Neighbors')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(title="Neighbors",bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.show()

success = np.array(success)
neighbour_success = success.sum(1)
best_neighbour = neighbour_success.argmax()
print(f'\nBest neighbour is  = {range(3,20)[best_neighbour]}')
# Ploting for the successes between different neighbour number
if plot:
    plt.title("Success rate with neighbours")
    plt.xlabel("Number of neighbours")
    plt.ylabel("Success Rate %")
    plt.ylim(0, 100)
    plt.plot(range(3, 20), neighbour_success/sum(tot)*100, marker="o")
    plt.show()
