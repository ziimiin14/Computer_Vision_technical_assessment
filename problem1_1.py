import cv2
import numpy as np

# Read the image
img = cv2.imread('./picture/problem1_1/1.jpg')

# Declare height,width, center of height, center of width and rotating angle
h,w = img.shape[:2]
center_w, center_h = w//2,h//2
angle = 5

# Obtain the rotation matrix with getRotationMatrix2D function and use wrapAffine function to map the rotated img to the initial width and heigh
rot_mat = cv2.getRotationMatrix2D((center_w,center_h),-angle,1.0)
rotated_img = cv2.warpAffine(img,rot_mat,(w,h))


# Cut out the desired part of the rotated image
cut_h = int(0.1 * h)
cut_w = int(0.1 * w )
rotated_img_cut = rotated_img[cut_h:(h-cut_h),cut_w:(w-cut_w),:]

# Finally, modify contrast of the picture using this equation--> alpha*img+beta
alpha = 1.4
beta = -40
rotated_img_cut_final = np.clip(alpha*rotated_img_cut+beta,0,255)
rotated_img_cut_final = rotated_img_cut_final.astype(np.uint8)

# Sharpen the photo
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
rotated_img_cut_final = cv2.filter2D(rotated_img_cut_final, -1, kernel)

# Save and show the iamge
cv2.imwrite('./result/problem1_1/Rotated+contrast_modified_image.jpg',rotated_img_cut_final)
cv2.imshow('Rotated+contrast modified image',rotated_img_cut_final )
cv2.waitKey(5000)
cv2.destroyAllWindows()