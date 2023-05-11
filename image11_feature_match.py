import numpy as np
import cv2

# Read the image file
img = cv2.imread("image11.jpg")
orange = cv2.imread("orange.jpg")
# cv2.imshow("Original Image", img)
# cv2.imshow("Orange", orange)
# Reduce size of output to 10% with preserve aspect ratio
img = cv2.resize(img, None, fx=0.1, fy=0.1)
orange = cv2.resize(orange, None, fx=0.1, fy=0.1)

#
result = cv2.matchTemplate(img, orange, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(max_val)
print(max_loc)

w = orange.shape[1]
h = orange.shape[0]
#cv2.rectangle(img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255),2)

threshold = .6
yloc, xloc = np.where(result>=threshold)
print(len(xloc))

# for (x,y) in zip(xloc, yloc):
#     cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,255),1)

cv2.imshow("Result", result)
cv2.imshow("Resized Image", img)
cv2.imshow("Resized Orange", orange)
cv2.waitKey(0)
