import cv2
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, imshow, waitKey
from numpy import zeros_like, ravel, sort, multiply, divide, int8


def median_filter(gray_img, mask=3):
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with median filter
    """
    # set image borders
    bd = int(mask / 2)
    # copy image size
    median_img = zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            # get mask according to mask
            kernel = ravel(gray_img[i - bd: i + bd + 1, j - bd: j + bd + 1])
            # calculate mask median
            median = sort(kernel)[int8(divide((multiply(mask, mask)), 2) + 1)]
            median_img[i, j] = median
    return median_img


def recognition(image, threshold):
    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(threshold, kernel, iterations=20)
    dilateW = cv2.dilate(threshold, kernel, iterations=6)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsW = cv2.findContours(dilateW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cntsW = cntsW[0] if len(cntsW) == 2 else cntsW[1]

    # Length of cnts array is equal num of rectangles we need
    i = len(cnts)
    iW = len(cntsW)



    # Looping through the array of contours
    for c in cnts:
        # (x, y) = pixels, w = width, h = height
        x, y, w, h = cv2.boundingRect(c)

        #Width and Height of the rectangle
        W = x + w
        H = y + h
        cv2.rectangle(image, (x, y), (W, H), (0, 0, 255), 3)

        # Calculating area of pixels
        p = 0
        words = 0

        for axes_x in range(x, W):
            for axes_y in range(y, H):
                if threshold[axes_y, axes_x] == 255:
                    p += 1

        for kappa in cntsW:
            xw, yw, ww, hw = cv2.boundingRect(kappa)
            if (xw < x and yw < y) and (xw+ww < W and  yw+hw < H):
                words += 1
                # Reading every word

        # Putting text for every rectangle that is created above
        cv2.putText(image, f'{i}', (x + 10, y + 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0, 0, 255), 4)
        print("ALL words "f'{i}', iW)
        print("Num of words "f'{i}', words)
        print("Boarding Area "f'{i}', w * h)
        print("Area "f'{i}', p, "\n")

        # Counting down the number of rectangles
        i = i - 1

    # Showing the results of dilated image
    cv2.namedWindow('dilate', cv2.WINDOW_NORMAL)
    cv2.imshow('dilate', dilate)
    cv2.waitKey(0)
    cv2.namedWindow('dilateW', cv2.WINDOW_NORMAL)
    cv2.imshow('dilateW', dilateW)
    cv2.waitKey(0)

    return image

def integral (image):
    for x in  range(image):
        print(x)
    return

if __name__ == "__main__":
    # Read original image
    img = imread("1_noise.png")
    imgOG = imread("1_original.png")
    # Turn image in gray scale value
    gray = cv2.imread("1_noise.png", cv2.IMREAD_GRAYSCALE)
    grayOG = cv2.imread("1_original.png", cv2.IMREAD_GRAYSCALE)

    # Get values with two different mask size
    # median3x3 = median_filter(gray, 3)

    # Show result images
    cv2.namedWindow('median filter with 3x3 mask', cv2.WINDOW_NORMAL)
    imshow("median filter with 3x3 mask", imgOG)
    cv2.waitKey(0)

    # Binary threshold
    _, thr1 = cv2.threshold(imgOG, 200, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('thresh_binary', cv2.WINDOW_NORMAL)
    imshow("thresh_binary", thr1)
    cv2.waitKey(0)

    # Converting color to gray and setting new threshold
    gray1 = cv2.cvtColo    
    # Starting the recognition
    #rectangle = recognition(imgOG, thresh)

    # Integral
    integ = integral(imgOG)

    # Showing the results of the image recognition
    cv2.namedWindow('results', cv2.WINDOW_NORMAL)
    cv2.imshow('results', integ)
    cv2.waitKey(0)
    waitKey(0)
