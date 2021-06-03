import numpy as np
import cv2
import os

path = './picture/problem1_2'
result_path = './result/problem1_2/'

images = os.listdir(path)
img_paths = [os.path.join(path,image) for image in images]


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
kernel = np.ones((5,5),np.uint8) # kernel for dilation

for img_path in img_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ## Image Preproscessing to detect the chessboard corner more easily

    # Create a mask(threshold) for chessboard in each picture
    ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    # Apply dilation to the mask so that the final output of the processed chessboard will have external white border 
    # for easier detection
    thresh_dil = cv2.dilate(thresh,kernel,iterations=2)

    # Apply the mask with the gray image
    gray_And = gray &thresh

    # Increase contrast and brightness to the chessboard mask
    location = np.where(gray_And>50)
    y ,x= location[0],location[1]
    gray_And[y[:],x[:]] =256- gray_And[y[:],x[:]]

    # Apply the white border mask to the chessboard in the gray image
    thresh_diff = thresh_dil- thresh
    gray_thresh = gray_And | thresh_diff



    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray_thresh, (8,6),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray_thresh,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        index = img_path[::-1].find('/')
        img_name = img_path[-index:]
        # Draw ,save and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners2, ret)
        cv2.imwrite(result_path+'detected_corner_'+img_name,img)
        cv2.imshow('Image with Detected corner', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

for img_path in img_paths:
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x,y,w,h = roi

    dst = dst[y:y+h,x:x+w]

    index = img_path[::-1].find('/')
    img_name = img_path[-index:]
    cv2.imwrite(result_path+'undistored_'+img_name,dst)
    cv2.imshow('Undistorted Image',dst )

    cv2.waitKey(500)

cv2.destroyAllWindows()


