#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import time

# Set camera model
pinhole_calibration = True

base_folder = 'mine'

file_entension = 'jpg'
#file_entension = 'png'

count = 0

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)

# For doorbell fisheye cameras
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
square_size = 2
objp = np.zeros((9*13,3), np.float32)

objp[:,:2] = np.mgrid[0:13,0:9].T.reshape(-1,2)
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(base_folder+'/_calibration/*.'+file_entension)

print ('Images are read')

# Chess_images
for fname in images:
    print (' Reading image =',count)
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.imshow('gray',gray)
    cv2.waitKey(100)

    if pinhole_calibration:
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (13,9),None)
        # ret, corners = cv2.findChessboardCorners(gray, (13,9),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)


    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (13,9), corners2,ret)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img',img)
        cv2.imwrite(base_folder+'/chess_images/Image'+str(count)+'.'+file_entension,img)
        count += 1
        cv2.waitKey(100)

# Printing for pinhole
if pinhole_calibration:
    # For calibrating normal camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print ('Pinhole Calibration results:')
    print ('mtx is:', mtx)
    print ('dist is:', dist)
    print ('rvecs is:', rvecs)
    print ('tvecs is:', tvecs)

images = glob.glob(base_folder+'/_undistortion/*.'+file_entension)

# Undistorting Images
l = 0
for fname2 in images:
    print ('undistorting images')
    img2 = cv2.imread(fname2)
    h,  w = img2.shape[:2]

    if pinhole_calibration:
        # Undistort for a normal camera
        newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
        # dst2 = cv2.undistort(img3, mtx, dist, None, newcameramtx)
        cv2.namedWindow('undistorted', cv2.WINDOW_NORMAL)
        cv2.imshow('undistorted',dst)
        cv2.waitKey(100)
        time.sleep(2.0)


    cv2.imwrite(base_folder+'/undistorted_images/undistorted_Image'+str(l)+'.'+file_entension,dst)
    l += 1

if pinhole_calibration:
    #### For normal camera calibration only
    mean_error = 0
    tot_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error
    print ('tot_error is:', tot_error, 'len(objpoints is:', len(objpoints))
    print ("mean_error is: ", tot_error/len(objpoints))


# Task2
def match2(d1, d2):
    n1 = d1.shape[0]
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
            matches.append(cv2.DMatch(i, i2, mindist2))

    return matches


images = glob.glob(base_folder+'/_stereo/*.'+file_entension)
surf = cv2.xfeatures2d_SURF.create(400)

kp = []
desc = []
l = 0
for fname3 in images:
    Simg = cv2.imread(fname3)
    h, w = Simg.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(Simg, mtx, dist, None, newcameramtx)
    kp[l] = surf.detect(dst)
    desc[l] = surf.compute(dst, kp[l])
    cv2.namedWindow('undistorted_stereo', cv2.WINDOW_NORMAL)
    cv2.imshow('undistorted_stereo', dst)
    cv2.waitKey(100)
    time.sleep(2.0)
    l += 1

    cv2.imwrite(base_folder+'/undistorted_stereo_img/undistorted_Image'+str(l)+'.'+file_entension,dst)
    print('Keypoint'+str(l)+' = ', kp[l], '\n')

match2(desc[0][1],desc[1][1])

cv2.destroyAllWindows()
