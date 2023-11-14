import cv2
import numpy as np

"""
INTEGRAL IMAGE
"""

filename = '../lesson3/a.jpg'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

int_img = cv2.integral(img)

# sum for (100, 100) - (200, 200)
sum = int_img[200, 200] - int_img[200, 100] - int_img[100, 200] + int_img[100, 100]
print(sum)
pass