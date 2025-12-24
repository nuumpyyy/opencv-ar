import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

def calibrate(width, height, dirpath, image_format, test_img):
    # initialize object points
    objp = np.zeros((width*height,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)
    
    # arrays to store object points and image points from images
    objpts = [] # 3d point in real world space
    imgpts = [] # 2d points in image plane
    
    images = glob.glob(dirpath + '/' + '*.' + image_format)
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # find chess board corners
        ret, corners = cv.findChessboardCorners(gray, (width,height), None)
    
        # if found, add object points and image points
        if ret == True:
            objpts.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpts.append(corners2)
    
    # obtain camera matrix, distortion coefficients, and rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)

    # refine camera matrix
    img = cv.imread(dirpath + '/' + test_img + '.' + image_format)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    return newcameramtx

in_mtx = calibrate(8, 6, 'calibration', 'jpg', '10')