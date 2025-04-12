import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from BOVW import vocabulary, bow_descs, img_paths, train_folders
plot = True
test_folders = ['caltech-101_5_test/accordion',
                'caltech-101_5_test/electric_guitar',
                'caltech-101_5_test/grand_piano',
                'caltech-101_5_test/mandolin',
                'caltech-101_5_test/metronome']
classes = np.array([x[20:] for x in train_folders])

sift = cv2.xfeatures2d_SIFT.create()

print('Training SVM...')
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))

# Train SVM
for i in range(0, len(classes)):
    if os.path.isfile(f'svm_{classes[i]}'):
        print(f'Picking up trained models')
    else:
        print(f'for {classes[i]}')
        labels = []
        for p in img_paths:
            if classes[i] in p:
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels, np.int32)
        svm.trainAuto(bow_descs, cv2.ml.ROW_SAMPLE, labels)
        svm.save(f'svm_{classes[i]}')
success = []
classes = np.array([x[19:] for x in test_folders])
# Load SVM
models = np.array([svm.load(f'svm_{classes[i]}') for i in range(0, len(classes))])

# Classification
descriptor_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)
tot = []
for folder in test_folders:
    files = os.listdir(folder)
    S_counter_class = 0
    SVM_PRE = np.zeros(len(classes))
    for test_img in files:
        path = os.path.join(folder, test_img)
        img = cv2.imread(path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        response = np.array([models[k].predict(bow_desc, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT) 
                            for k in range(0, len(classes))], np.float32)[:, 1]
        pr_id = response.argmin()
        # print(f'It is a {pr_id}')
        
        if pr_id == test_folders.index(folder):
            S_counter_class += 1
            # print(f'It is a {classes[pr_id]}')
        
    tot.append(len(files))
    success.append(S_counter_class)
    print(f'Successes of {classes[test_folders.index(folder)]} = {S_counter_class}/{len(files)}')
print("\n====Success Rates====")
for i in range (0,len(classes)):
    print(f'Success rate of {classes[i]} = {round(success[i]/tot[i] *100, 2)}%')
print(f'Success rate of SVM = {round(sum(success)/sum(tot)*100,2)}%')
if plot:
    plt.bar(classes, [a/b for a, b in zip(success, tot)])

    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate of Each Class')

    # Show the plot
    plt.show()
