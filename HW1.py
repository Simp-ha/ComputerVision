import cv2
import numpy as np
from cv2 import imread, imshow, waitKey
from numpy import zeros_like, ravel, sort, multiply, divide, int8


def median_filter(gray_img):
    # set image borders
    bd = int(3/2)
    # copy image size
    median_img = zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            # Building kernel
            kernel = ravel(gray_img[i - bd: i + bd + 1, j - bd: j + bd + 1])
            # Calculate median mask
            median = sort(kernel)[int8(divide((multiply(3, 3)), 2) + 1)]
            median_img[i, j] = median
    return median_img


def integral(image):
    # Creating table with the same dimension
    rows, columns =  image.shape
    integ_img = np.zeros((rows+1, columns+1), dtype=image.dtype).astype(float)
    for x in range(1,len(integ_img)):
        for y in range(1, len(integ_img[0])):
            integ_img[x, y] = image[x-1, y-1] + integ_img[x, y-1] + integ_img[x-1, y] - integ_img[x-1, y-1]
    return integ_img


def recognition(image, threshold):

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(threshold, kernel, iterations=20)
    dilateW = cv2.dilate(threshold, kernel, iterations=5)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsW = cv2.findContours(dilateW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cntsW = cntsW[0] if len(cntsW) == 2 else cntsW[1]

    # Length of cnts array is equal num of rectangles we need
    i = len(cnts)
    iW = len(cntsW)

    # Integral Image
    grayConv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    integral_image = integral(grayConv)

    # Looping through the array of contours
    for c in cnts:
        # (x, y) = pixels, w = width, h = height
        x, y, w, h = cv2.boundingRect(c)

        # Width and Height of the rectangle
        W = x + w
        H = y + h
        cv2.rectangle(image, (x, y), (W, H), (0, 0, 255), 3)

        # Integral Image
        MeanValue = (integral_image[y, x] + integral_image[H, W] - integral_image[y, W] - integral_image[H, x]) / (w*h)

        # Counting Pixels of words
        p = 0
        for axes_x in range(x, W):
            for axes_y in range(y, H):
                if threshold[axes_y, axes_x] == 255:
                    p += 1

        # Counting words
        words = 0
        for kappa in cntsW:
            xw, yw, ww, hw = cv2.boundingRect(kappa)
            if xw >= x and yw >= y and xw + ww <= W and yw + hw <= H:
                words += 1

        # Putting text for every rectangle that is created above
        cv2.putText(image, f'{i}', (x + 10, y + 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0, 0, 255), 4)
        print("-------Region "f'{i}-------')
        print("ALL words: ", iW)
        print("Num of words: ", words)
        print("Boarding Area: ", w * h)
        print("Area: ", p)
        print("Mean gray-level value in bounding box: ", MeanValue,"\n")

        # Counting down the number of rectangles
        i = i - 1

    # Showing the results of dilated image
    cv2.namedWindow('dilate', cv2.WINDOW_NORMAL)
    cv2.imshow('dilate', dilate)
    cv2.waitKey(0)
    cv2.namedWindow('dilateW', cv2.WINDOW_NORMAL)
    cv2.imshow('dilateW', dilateW)
    cv2.waitKey(0)

    return 0;


if __name__ == "__main__":
    # Read original image
    for i in range(2, 6):
        img = imread(f'{i}'"_noise.png")
        imgOG = imread(f'{i}'"_original.png")

        # Turn image in gray scale value
        gray = cv2.imread(f'{i}'"_noise.png", cv2.IMREAD_GRAYSCALE)
        # grayOG = cv2.imread(f'{i}'"_original.png", cv2.IMREAD_GRAYSCALE)

        # Get values with two different mask size
        median = median_filter(gray)

        # Show result images
        cv2.namedWindow(f'{i}''Median filter', cv2.WINDOW_NORMAL)
        imshow(f'{i}'"Median filter", median)
        cv2.waitKey(0)

        # Binary threshold
        _, thr = cv2.threshold(imgOG, 200, 255, cv2.THRESH_BINARY)
        cv2.namedWindow(f'{i}''thresh_binary', cv2.WINDOW_NORMAL)
        imshow(f'{i}'"thresh_binary", thr)
        cv2.waitKey(0)

        # Converting color to gray and setting new threshold
        grayConv = cv2.cvtColor(imgOG, cv2.COLOR_BGR2GRAY)
        _, thr2 = cv2.threshold(grayConv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.namedWindow(f'{i}''thresh_binary', cv2.WINDOW_NORMAL)
        imshow(f'{i}'"thresh_binary", thr2)
        cv2.waitKey(0)

        # Recognition of text and making rectangles around them
        rectangle = recognition(imgOG, thr2)

        waitKey(0)