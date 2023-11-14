import cv2
import numpy as np

def sobel_mag(img):
    img1 = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    img2 = cv2.Sobel(img, cv2.CV_32F, 1, 0)

    img3 = np.sqrt(np.power(img1, 2) + np.power(img2, 2))
    return img3.astype(np.uint8)


img = cv2.imread('cells.png', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)
gradient_mag = sobel_mag(img)

cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow('original', img)

cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges', edges)

cv2.namedWindow('grad', cv2.WINDOW_NORMAL)
cv2.imshow('grad', gradient_mag)

cv2.waitKey(0)
pass