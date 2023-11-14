import cv2
import numpy as np

"""
MORPHOLOGICAL TRANSFORMATIONS
"""

filename = 'cells.png'
#filename = 'j_salt.png'
# filename = 'j_pepper.png'

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('main', )
cv2.imshow('main', img)
cv2.waitKey(0)

# strel = np.ones((5,5), np.uint8)
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# EROSION
erode = cv2.morphologyEx(img, cv2.MORPH_ERODE, strel)
cv2.namedWindow('calc', )
cv2.imshow('calc', erode)
cv2.waitKey(0)

# DILATION
dilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, strel)
cv2.namedWindow('calc', )
cv2.imshow('calc', dilate)
cv2.waitKey(0)

# OPENING
# open = cv2.morphologyEx(img, cv2.MORPH_OPEN, strel)
# cv2.namedWindow('calc', )
# cv2.imshow('calc', open)
# cv2.waitKey(0)

# CLOSING
# close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, strel)
# cv2.namedWindow('calc', )
# cv2.imshow('calc', close)
# cv2.waitKey(0)

# #OPENING MANUAL
open = cv2.morphologyEx(erode, cv2.MORPH_DILATE, strel)
cv2.namedWindow('calc2', )
cv2.imshow('calc2', open)
cv2.waitKey(0)

# #CLOSING MANUAL
close = cv2.morphologyEx(dilate, cv2.MORPH_ERODE, strel)
cv2.namedWindow('calc2', )
cv2.imshow('calc2', close)
cv2.waitKey(0)

# EDGE
gradient = dilate - erode
cv2.namedWindow('calc', )
cv2.imshow('calc', gradient)
cv2.waitKey(0)

cv2.destroyAllWindows()
