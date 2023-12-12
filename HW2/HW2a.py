import glob

import cv2
import numpy as np
import cv2 as cv

sift = cv.xfeatures2d_SIFT.create(400)


def panorama(img1, img2):
    cv.namedWindow('main1')
    cv.imshow('main1', img1)
    cv.waitKey(0)

    kp1 = sift.detect(img1)
    desc1 = sift.compute(img1, kp1)

    cv.namedWindow('main2')
    cv.imshow('main2', img2)
    cv.waitKey(0)

    kp2 = sift.detect(img2)
    desc2 = sift.compute(img2, kp2)

    def match2(d1, d2):
        n1 = d1.shape[0]
        n2 = d2.shape[0]

        matches = []
        for i in range(n1):
            fv = d1[i, :]
            diff = d2 - fv
            diff = np.abs(diff)
            distances = np.sum(diff, axis=1)

            i2 = np.argmin(distances)
            mindist2 = distances[i2]

            distances[i2] = np.inf

            i3 = np.argmin(distances)
            mindist3 = distances[i3]

            if mindist2 / mindist3 < 0.5:
                matches.append(cv.DMatch(i, i2, mindist2))

        return matches

    matches = match2(desc1[1], desc2[1])

    dimg = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches, None)
    cv.namedWindow('main3')
    cv.imshow('main3', dimg)
    cv.waitKey(0)

    img_pt1 = []
    img_pt2 = []
    for x in matches:
        img_pt1.append(kp1[x.queryIdx].pt)
        img_pt2.append(kp2[x.trainIdx].pt)
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)
    # Βρίσκει πώς πρέπει να μετατραπει η πρώτη για να "ταιριάξει" με τη δευτερη

    img3 = cv.warpPerspective(img2, M, (img1.shape[1]+200, img1.shape[0]+200))
    img3[0: img1.shape[0], 0: img1.shape[1]] = img1
    return img3

# images = glob.glob('/rio-01.png')
img = [cv2.imread("rio-01.png"), cv2.imread("rio-02.png"), cv2.imread("rio-03.png"), cv2.imread("rio-04.png")]

stitch1 = panorama(img[0], img[1])
cv.imwrite('rio12.png', stitch1)
cv.namedWindow('main')
cv.imshow('main', stitch1)
cv.waitKey(0)

panorama(img[2], img[3])
stitch2 = panorama(img[2], img[3])
cv.imwrite('rio34.png', stitch2)
cv.namedWindow('main')
cv.imshow('main', stitch2)
cv.waitKey(0)

fin_panorama = panorama(stitch1, stitch2)
cv.imwrite('rio.png', fin_panorama)
cv.namedWindow('main')
cv.imshow('main', fin_panorama)
cv.waitKey(0)


pass

