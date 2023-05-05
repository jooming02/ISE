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
blue_bgnd[:,:,0]=255 # blue color
# Copy each object in the image to blue background
cv2.copyTo(img_resized,mask,blue_bgnd)
cv2.imshow("Background", blue_bgnd)

# Import a background image
table_bgnd = cv2.imread("background.jpg")
height, width = img_resized.shape[:2]
table_bgnd = cv2.resize(table_bgnd, (width, height))
# Copy each object in the image to background image
cv2.copyTo(img_resized,mask,table_bgnd)
cv2.imshow("BG", table_bgnd)

N, idx, stats, cent = cv2.connectedComponentsWithStats(bw)
print("Number of connected components : ",N)
print(" Indices : ",np.unique(idx))
# Declare the counter as 0
cnt = 0
for s in stats:
    x = s[0]
    y = s[1]
    w = s[2]
    h = s[3]
    #print(w)
    if w > 10 and w < 106:
        cnt += 1
        cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 0, 255), 3)

# Contouring
cont, hier = cv2.findContours(bw, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

contour = []
for co in cont:
    #print("Area : ", cv2.contourArea(co), ", ---- Perimeter : ", cv2.arcLength(co, True))
    area = cv2.contourArea(co)
    eps = 0.04 * cv2.arcLength(co, True)
    approx_cont = cv2.approxPolyDP(co, eps, True)
    if area > 1200:
        contour.append(len(approx_cont))
        cv2.drawContours(img_contour, [approx_cont], -1, (255, 0, 0), 5)

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

print("Number of triangles: ", triangle)
print("Number of rectangles: ", rectangle)
print("Number of circles: ", circle)
print("Number of objects: ", cnt)
#cv2.imshow("contour", img_contour)
cv2.imshow("original", img_resized)
cv2.imshow("BlackWhite", bw)
#cv2.imshow("GreyScale", grey)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
