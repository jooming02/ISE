import cv2
import numpy as np

img = cv2.imread("image1.jpg")
# Reduce size of output to 10% with preserve aspect ratio
img_resized = cv2.resize(img, None, fx=0.1, fy=0.1)
# create a new copy of resized image for contouring
img_contour = np.copy(img_resized)
# Convert to grayscale image
grey = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# Convert to binary image
r, bw = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)

################################################ OBJECT COLOR START ################################
# create a new copy of resized image for check color
img_color = np.copy(img_resized)
# Convert to HSV color space
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

# Define the range of color in HSV
lower_red = np.array([0, 255, 255])
upper_red = np.array([10, 255, 255])

lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the image to get a binary mask of the orange pixels
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_green, upper_green)
mask3 = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply a morphological closing operation to merge nearby orange circles
kernel = np.ones((21, 21), np.uint8)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Draw the contours on the original image
def checkArea(contour, colour):
    color = 0
    for co in contour:
        area = cv2.contourArea(co)
        if area > 1000:
            cv2.drawContours(img_color, contour, -1, colour, 2)
            color += 1
    return color


red = checkArea(contours1, (0, 0, 255))
green = checkArea(contours2, (0, 255, 0))
blue = checkArea(contours3, (255, 0, 0))
################################################ OBJECT COLOR END ################################


# Kernel for morphology
morp_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Apply blurring effect
mask = cv2.GaussianBlur(bw, (5, 5), 2)
# Close the gaps
mask = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel=morp_ker, iterations=1)
# Fill the region with white color
cv2.floodFill(mask, None, (204, 80), (255, 0, 0))
cv2.floodFill(mask, None, (204, 40), (255, 0, 0))

# Create a blue background
blue_bgnd = np.zeros_like(img_resized)
blue_bgnd[:, :, 0] = 255  # blue color
# Copy each object in the image to blue background
cv2.copyTo(img_resized, mask, blue_bgnd)
cv2.imshow("Blue Background", blue_bgnd)

# Import a background image
table_bgnd = cv2.imread("background.jpg")
height, width = img_resized.shape[:2]
table_bgnd = cv2.resize(table_bgnd, (width, height))
# Copy each object in the image to background image
cv2.copyTo(img_resized, mask, table_bgnd)
cv2.imshow("Change Background", table_bgnd)

N, idx, stats, cent = cv2.connectedComponentsWithStats(bw)
print("Number of connected components : ", N)
print(" Indices : ", np.unique(idx))
# Declare the counter as 0
cnt = 0
for s in stats:
    x = s[0]
    y = s[1]
    w = s[2]
    h = s[3]
    # print(w)
    if w > 10 and w < 106:
        cnt += 1
        cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 0, 255), 3)

# Contouring
cont, hier = cv2.findContours(bw, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

contour = []
i=0
for co in cont:
    # print("Area : ", cv2.contourArea(co), ", ---- Perimeter : ", cv2.arcLength(co, True))
    area = cv2.contourArea(co)
    eps = 0.04 * cv2.arcLength(co, True)
    approx_cont = cv2.approxPolyDP(co, eps, True)
    if area > 1200:
        contour.append(len(approx_cont))
        cv2.drawContours(img_contour, [approx_cont], -1, (255, 0, 0), 5)
        M = cv2.moments(co)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(img_contour, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i=i+1;

triangle = 0
rectangle = 0
circle = 0
for i in contour:
    if i == 3:
        triangle += 1
    elif i == 4:
        rectangle += 1
    else:
        circle += 1

print("Number of objects: ", cnt)
print("Number of triangles: ", triangle)
print("Number of rectangles: ", rectangle)
print("Number of circles: ", circle)
print("Number of red objects:" + str(red))
print("Number of green objects:" + str(green))
print("Number of blue objects:" + str(blue))
print("Number of other color objects:" + str(cnt - blue))

cv2.imshow("contour", img_contour)
cv2.imshow("Detect Object", img_resized)
# cv2.imshow("BlackWhite", bw)
# cv2.imshow("GreyScale", grey)
# cv2.imshow("Mask", mask)
# Display the result
cv2.imshow('Number of different colors object (Result)', img_color)

cv2.waitKey(0)
