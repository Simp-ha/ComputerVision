import os.path
import cv2
import numpy as np
from cv2 import imread
from numpy import zeros_like, ravel, sort, multiply, divide, int8
AM = "58401"
# Saving results
sv = True

def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name


# Monitor the results of Gaussian
def monitor(win, save=True):
    for i in range(0, len(win)):
        var_name = get_var_name(win[i])
        if var_name is None:
            cv2.namedWindow(f'{i}', cv2.WINDOW_NORMAL)
            cv2.imshow(f'{i}', win[i])
        else:
            cv2.namedWindow(var_name, cv2.WINDOW_NORMAL)
            cv2.imshow(var_name, win[i])
        if save and var_name is not None:
            cv2.imwrite(var_name + ".png", win[i])
        else:
            cv2.imwrite(f'{i}'+".png", win[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def median_filter(gray_img):
    # set image borders
    bd = int(3 / 2)
    # copy image size
    x = gray_img.shape[0]-bd
    y = gray_img.shape[1]-bd
    median_img = zeros_like(gray_img)
    for i in range(bd, x):
        for j in range(bd, y):
            # Building kernel
            kernel = ravel(gray_img[i - bd: i + bd + 1, j - bd: j + bd + 1])
            # Calculate median mask
            median_img[i, j] = sort(kernel)[int8(divide((multiply(bd, bd)), 2))]
    median_img[0] += gray_img[0]
    median_img[x] += gray_img[x]
    median_img.transpose()[0] += gray_img.transpose()[0]
    median_img.transpose()[y] += gray_img.transpose()[y]
    return median_img


def gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def integral(image):
    # Creating table with the same dimension
    rows, columns = image.shape
    integ_img = np.zeros((rows + 1, columns + 1), dtype=image.dtype).astype(float)
    for x in range(1, len(integ_img)):
        for y in range(1, len(integ_img[0])):
            integ_img[x, y] = image[x - 1, y - 1] + integ_img[x, y - 1] + integ_img[x - 1, y] - integ_img[x - 1, y - 1]
    return integ_img


def recognition(image, threshold):
    # Create rectangular structuring element and dilate  ----LINES
    Lkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilateL = cv2.dilate(threshold, Lkernel, iterations=15)

    # Create rectangular structuring element and dilate  ----WORDS
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilateW = cv2.dilate(threshold, kernel, iterations=15)

    # Find contours and draw rectangle ----LINES
    cntsL = cv2.findContours(dilateL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsL = cntsL[0] if len(cntsL) == 2 else cntsL[1]

    # Find contours and draw rectangle ----WORDS
    cntsW = cv2.findContours(dilateW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsW = cntsW[0] if len(cntsW) == 2 else cntsW[1]

    # Integral Image
    grayC = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    integral_image = integral(grayC)
    f = open("Results.txt", "a")
    f.write("\n\n==============" + get_var_name(image) + "==============")
    # Looping through the array of contours
    for ll, c in enumerate(cntsL[::-1], start=1):
        # (x, y) = pixels, w = width, h = height
        x, y, w, h = cv2.boundingRect(c)

        # Width and Height of the rectangle
        W = x + w  # LAST PIXEL OF CONTOUR IN A LINE
        H = y + h  # LAST PIXEL OF CONTOUR IN A COLUMN
        cv2.rectangle(image, (x, y), (W, H), (0, 0, 255), 4)

        # Integral Image
        MeanValue = (integral_image[y, x] + integral_image[H, W]
                     - integral_image[y, W] - integral_image[H, x]) / (w * h)

        # Counting Pixels of words
        p = 0
        for axes_y in range(y, H):
            for axes_x in range(x, W):
                if threshold[axes_y, axes_x] == 255:
                    p += 1

        # Counting words
        words = 0
        for kappa in cntsW:
            xw, yw, ww, hw = cv2.boundingRect(kappa)
            if (xw >= x and yw >= y) and (ww + xw <= W and hw + yw <= H):
                words += 1
            # cv2.rectangle(image, (xw, yw), (xw + ww, yw + hw), (0, 255, 0), 2)

        string = ("\n-------Region "f'{ll}'"-------"
                  "\nArea(px): " + str(p) +
                  "\nBounding Box Area(px): " + str(w * h) +
                  "\nNum of words: " + str(words) +
                  "\nMean gray-level value in bounding box: " + str(MeanValue) + "\n")

        f.write(string)
        print(string)

        # Putting text for every rectangle that is created above
        cv2.putText(image, f'{ll}', (x, y+100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0, 0, 255), 8)
        # Counting down the number of rectangles

    f.close()


    # Showing the results of dilated image
    monitor([dilateW, dilateL, image], sv)
    return 0


if __name__ == "__main__":

    # Read original image
    img = imread("HW1/"f'{AM[4]}'".png")

    # Gaussian noise input
    Gauss = gaussian_noise(img)
    grayConvB4 = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
    if not os.path.exists("Median.jpg"):
        median = median_filter(grayConvB4)
        cv2.imwrite("Median.jpg", median)
        # bad code
    median = imread("Median.jpg")

    # Converting color to gray and setting the threshold for the noisy pic
    grayConvM = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(grayConvM, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Converting color to gray and setting threshold for original
    grayConv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thr2 = cv2.threshold(grayConv, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    monitor([img, Gauss, grayConvB4, median, thr, thr2], sv)

    # Noisy threshold
    # grayConvM = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
    # _, thr_noise = cv2.threshold(grayConvM, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # monitor([img, Gauss, thr_noise], sv)
    # recognition(Gauss, thr_noise)

    # # Recognition of text and making rectangles around them
    recognition(median, thr)
    recognition(img, thr2)
