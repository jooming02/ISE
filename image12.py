import numpy as np
import cv2

# Read the image file
img = cv2.imread("image12.jpg")
orange = cv2.imread("orange.jpg")
# Reduce size of output to 10% with preserve aspect ratio
img = cv2.resize(img, None, fx=0.1, fy=0.1)
orange = cv2.resize(orange, None, fx=0.1, fy=0.1)

# Template Matching to find the orange in the image
result = cv2.matchTemplate(img, orange, cv2.TM_CCOEFF_NORMED)
# Find the location and coefficient of worst and best matched image
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Result
print("Minimum Coefficnet: ",min_val)
print("Minimum Location: ",min_loc)
print("Maximum Coefficnet: ",max_val)
print("Maximum Location: ",max_loc)

width_orange = orange.shape[1]
height_orange = orange.shape[0]

# Find multiple object with 60% similarity
threshold = .60
yloc, xloc = np.where(result>=threshold)

for (x,y) in zip(xloc, yloc):
    cv2.rectangle(img, (x,y), (x + width_orange,y + height_orange),(0,255,255),1)

cv2.imshow("Result", result)
cv2.imshow("Resized Image", img)
cv2.imshow("Resized Orange", orange)
cv2.waitKey(0)
