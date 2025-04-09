import cv2
import numpy as np
import glob
import time
Testing = False
file_extension = 'jpg'
folders = ['GES-50', 'NISwGSP', 'OpenPano/flower', 'mine']
# This for asking before
yes = input("Please write which panorama do you want\n1)GES\n2)NIS\n3)Flower\n4)Mine\nYour Answer is: ")

# Importing images
images = glob.glob(folders[int(yes) - 1] + "/*." + file_extension)

# Creating SIFT object
sift = cv2.xfeatures2d_SIFT.create(400)
# Creating SURF object
surf = cv2.xfeatures2d_SURF.create(400)
algorithms = [sift, surf]
NAMES = ['SIFT', 'SURF']
img = []
imgs = []
panorama = []


def match(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]
    matches = []
    for featureIdx in range(n1):
        # Selecting every feature
        fv = d1[featureIdx, :]
        diff = np.abs(d2 - fv)
        distances = np.sum(diff, axis=1)
        # Finding the least distance between feature from d1 and d2
        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        # Same but the second least distance
        distances[i2] = np.inf
        i3 = np.argmin(distances)
        mindist3 = distances[i3]

        if mindist2 / mindist3 < 0.5:
            # Every feature with mindist2 < half of mindist3 is stored with the
            # queryIdx, trainIdx and distance between the 2
            matches.append(cv2.DMatch(featureIdx, i2, mindist2))
    return matches


def find_everything(img_idx):
    # Matching method
    d1 = descr1[1]
    d2 = descr2[1]
    match_start = time.time()
    matches = match(d1, d2)
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # matches = bf.match(d1, d2)
    match_end = time.time()
    file1.write(f"MATCHES: {len(matches)}\nTime spent: {(match_end-match_start)} sec \tImages: {img_idx} - {img_idx+1} \t"
                f"Folder: {folders[int(yes)-1]}\n")


    # # Cross-checking not working can't find same matches
    # m1 = match(d1, d2)
    # m2 = match(d2, d1)
    # for x in m2:
    #     for y in m1:
    #         if x == y:
    #             matches.append(m1)

    # Drawing the matches on the image
    if Testing:
        dimg = cv2.drawMatches(img[img_idx], kp1, img[img_idx + 1], kp2, matches, None)
        cv2.namedWindow('D', cv2.WINDOW_NORMAL)
        cv2.imshow('D', dimg)
        cv2.waitKey(0)

    # It stores only the keypoints with the matched descriptors.
    # We make it array, so they can be read by the Homography method
    img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches])
    img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches])

    # Finding the Homography matrix (map M) for fitting the image_pt2 into img_pt1 according to RANSAC algorithm
    M1, _ = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)
    rows, columns = img[img_idx].shape[:2]
    # columns = max(columns2, columns1)
    img31 = cv2.warpPerspective(img[img_idx + 1], M1, (2*columns, 2*rows))  # (x = columns, y = rows)

    # Exporting the results of the wrapped image
    # cv2.imwrite(f'{NAMES[algorithms.index(s)]}_Wrapped{t}.jpg', img31)
    if Testing:
        cv2.namedWindow('W', cv2.WINDOW_NORMAL)
        cv2.imshow('W', img31)
        cv2.waitKey(0)

    # Exporting the results of the cropped-wrapped image
    # cv2.imwrite(f'{yes}/Wrapped-Cropped{img_idx}.jpg', img31[0:r, 0:c])

    # Another way
    # result = np.copy(img31)
    # result[0:rows, 0:columns] = img[img_idx]

    result = np.copy(img31)
    black_mat = [0, 0, 0]
    mask = np.all(result[:rows, :columns] == black_mat, axis=-1)
    result[:rows, :columns][mask] = img[img_idx][mask]

    # Binding the first image at the first half pixels of the output
    # result[0:rows, 0:columns] = img[img_idx]
    # result[:, columns // 2:] = img31[:, columns // 2:]

    # Rows and columns of the last non-black pixels
    NBPixels = np.argwhere(result > 0)
    r = max(NBPixels[:, 0])
    c = max(NBPixels[:, 1])
    result = result[0:r, 0:c]

    # Results
    cv2.imwrite(f'{yes}/{NAMES[algorithms.index(s)]}_result_{i}_{i+1}.jpg', result)
    if Testing:
        cv2.namedWindow(f'main{i}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'main{i}', result)
        cv2.waitKey(0)
    return result


starting = time.time()
# Inserting every photo
for i in images:
    imgs.append(cv2.imread(i))
# Reading the photos for every algorithm
s = sift
# for s in algorithms:
img = imgs.copy()
file1 = open("TIMES.txt", "a")
file1.write(f"{NAMES[algorithms.index(s)]} \n")
for j in range(0, len(img)-2):
    if len(img) % 2 > 0:
        N = len(img)-1
        A_img = img[N]
    else:
        N = len(img)
        A_img = np.array([])

    for i in range(0, N, 2):
        print(f'Compute {i} batch of {NAMES[algorithms.index(s)]}')
        kp1 = (s.detect(img[i]))
        descr1 = (s.compute(img[i], kp1))

        kp2 = (s.detect(img[i+1]))
        descr2 = (s.compute(img[i+1], kp2))

        panorama.append(find_everything(i))
    img = panorama.copy()
    panorama.clear()
    cv2.destroyAllWindows()
    if np.any(A_img):
        img.append(A_img)
ending = time.time()
print(f"It took {(ending - starting)} seconds")
file1.write(f"Folder: {str(folders[int(yes) - 1])}\tExTime: {(ending - starting)} seconds\n")
file1.close()
img.clear()
starting = time.time()
