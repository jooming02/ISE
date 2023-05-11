import cv2
import numpy as np

#denoise
img = cv2.imread("image2.jpg")
img_obj = img
img_copy = img
img_bg = img

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r, c = cv2.threshold(grey, 110, 255, cv2.THRESH_BINARY)
# ker = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]])
ker = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 25, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])


msk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


img2 = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel=msk, iterations=10)
img2 = cv2.medianBlur(img2, 21)
# img2 = cv2.GaussianBlur(img2, (13, 13), 5)
# img2 = cv2.filter2D(img2, -1, ker)
img2 = cv2.resize(img2, None, fx=0.1, fy=0.1)
grey = cv2.resize(grey, None, fx=0.1, fy=0.1)
img = cv2.resize(img, None, fx=0.1, fy=0.1)
# img_copy = cv2.resize(img_copy, None, fx=0.1, fy=0.1)
c = cv2.resize(c, None, fx=0.1, fy=0.1)
################################################ find shape,total and size Start ################################
N, idx, stats, cent = cv2.connectedComponentsWithStats(img2)
counter = 0

for s in stats:
    x = s[0]
    y = s[1]
    w = s[2]
    h = s[3]
    
    if w > 40 and w < 120:
        counter += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        img_copy[y:y+h, x:x+w, :] = 0

#Contouring
cont,hier = cv2.findContours(img2,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)

contour = []
for co in cont:
    print("Area : ", cv2.contourArea(co), ", ---- Perimeter : ",cv2.arcLength(co,True))
    area = cv2.contourArea(co)
    eps=0.06*cv2.arcLength(co,True)
    approx_cont = cv2.approxPolyDP(co,eps,True)
    if area > 1500:
        contour.append(len(approx_cont))
        
        cv2.drawContours(img,[approx_cont],-1,(255,0,0),5)

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

grey2 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

r2, c2 = cv2.threshold(grey2, 100, 255, cv2.THRESH_BINARY)
c2 = cv2.resize(c2, None, fx=0.1, fy=0.1)
print("Number of triangles: ", triangle)
print("Number of rectangles: ", rectangle)
print("Number of circles: ", circle)
print("Number of objects: ", counter)

################################################ OBJECT COLOR START ################################
# create a new copy of resized image to count the total number of object in different colour.
img_color = np.copy(img)
# Convert to HSV color space
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
# Convert RGB color code to BGR color code
rgb_color = np.uint8([[[4, 235, 241]]])
bgr_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2BGR)

# Convert BGR color code to HSV color space
hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
# Define the range of color in HSV
lower_cyan = np.array([hsv_color[0][0][0]-10, 50, 50])
upper_cyan = np.array([hsv_color[0][0][0]+10, 255, 255])

# Threshold the image to get a binary mask of the colour pixels
mask1 = cv2.inRange(hsv, lower_cyan, upper_cyan)


# Apply a morphological closing operation to merge the nearby same colour area
kernel = np.ones((7, 7), np.uint8)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=3)


# Find contours in the mask
contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Check color on the image
def checkColor(contour):
    color = 0
    for co in contour:
        area = cv2.contourArea(co)
        eps=0.01*cv2.arcLength(co,True)
        approx_cont = cv2.approxPolyDP(co,eps,True)
        print(area)
        if area >500 and area < 5000:
            cv2.drawContours(img_color,[approx_cont],-1,(255,0,0),5)
            color += 1
    return color


# check object with color Red, green, and blue colour
cyan = checkColor(contours1)
print(cyan)

################################################ OBJECT COLOR END ################################
################################################ Extract One By One Start ################################

# Contouring
cont, hier = cv2.findContours(c, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

result = []

# create a new copy of resized image to extract object one by one
img_obj = cv2.resize(img_obj, None, fx=0.1, fy=0.1)

#draw contour one by one in new img_obj
for co in cont:
    area = cv2.contourArea(co)
    if area > 1500:
        mask = np.zeros(img_obj.shape[:2], np.uint8)
        cv2.drawContours(mask, [co], -1, 255, -1)
        result.append(cv2.bitwise_and(img_obj, img_obj, mask=mask))

i = 0
while i < len(result):
    # Display each object in array result one by one
    cv2.imshow("object" + str(i + 1), result[i])
    i = i + 1

################################################ One By One End ################################

################################################ Change BackGround Start ################################
# cv2.imshow("BlackWhite", bw)
# Close the gaps

morp_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel=morp_ker, iterations=1)
#cv2.imshow("Morphology Close", mask)

# Fill the region with white color
cv2.floodFill(mask, None, (385, 45), 0)


# cv2.floodFill(mask, None, (204, 40), (255, 0, 0))
#cv2.imshow("Mask", mask)

# Create a blue background
img_bg = cv2.resize(img_bg, None, fx=0.1, fy=0.1)
blue_bgnd = np.zeros_like(img_bg)
blue_bgnd[:, :, 0] = 255  # blue color
# Copy each object in the image to blue background
cv2.copyTo(img_bg, mask, blue_bgnd)

# Import a background image
table_bgnd = cv2.imread("background.jpg")
height, width = img_bg.shape[:2]
table_bgnd = cv2.resize(table_bgnd, (width, height))
# Copy each object in the image to background image
cv2.copyTo(img_bg, mask, table_bgnd)

cv2.imshow("Blue Background", blue_bgnd)
cv2.imshow("Table Background", table_bgnd)

################################################ Change BackGround End ################################

cv2.imshow("original", img)
cv2.imshow("black", c)
cv2.imshow("ori", grey)
cv2.imshow("lena2", img2)
cv2.imshow("img_color", img_color)
cv2.imshow("img_copy", img_copy)
cv2.imshow("c2", c2)
cv2.imshow("mask", mask)
cv2.waitKey(0)
