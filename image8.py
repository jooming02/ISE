import cv2
import numpy as np

# Read the image file
img = cv2.imread("image8.jpg")
# Reduce size of output to 10% with preserve aspect ratio
img_resized = cv2.resize(img, None, fx=0.1, fy=0.1)
# cv2.imshow("Original Image", img)
# cv2.imshow("Resized Image", img_resized)

# Brighten the image
brightness = 1.8
img_bright = cv2.convertScaleAbs(img_resized, alpha=brightness, beta=0)
cv2.imshow("Bright Image ",img_bright)
# Convert to grayscale image
grey = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# Convert to binary image
r, bw = cv2.threshold(grey, 20, 255, cv2.THRESH_BINARY)
cv2.imshow("BlackWhite", bw)
# Kernel for morphology
morp_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Close the gaps
mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel=morp_ker, iterations=1)
cv2.imshow("Morphology Close", mask)

################################################ find shape,total and size Start ################################
# create a new copy of resized image to count the total number of object in the image
img_counting = np.copy(img_bright)
# create a new copy of resized image to count the total number of object in different shape
img_shape = np.copy(img_bright)

N, idx, stats, cent = cv2.connectedComponentsWithStats(mask)
# print("Number of connected components : ", N)
# print(" Indices : ", np.unique(idx))

# Declare the counter as 0
cnt = 0
i=0
for s in stats:
    x = s[0]
    y = s[1]
    width = s[2]
    height = s[3]
    #print(width)
    if width > 4 and width < 300:
        cnt += 1
        cv2.rectangle(img_counting, (x, y), (x + width, y + height), (0, 0, 255), 3)
        w = int(width / 2)
        h = int(height / 2)
        cv2.putText(img_counting, str(i + 1), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        i = i + 1

# Contouring
cont, hier = cv2.findContours(mask, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

contour = []

max_area = 0
min_area = float('inf')
largest = None
smallest = None

for co in cont:
    #print("Area : ", cv2.contourArea(co), ", --- Perimeter : ", cv2.arcLength(co, True))
    area = cv2.contourArea(co)
    eps = 0.04 * cv2.arcLength(co, True)
    approx_cont = cv2.approxPolyDP(co, eps, True)
    # Get Rid of the area of the hole
    if area > 100:
        contour.append(len(approx_cont))
        cv2.drawContours(img_shape, [approx_cont], -1, (255, 0, 0), 5)

        # find the largest object
        if area > max_area:
            max_area = area
            largest = co
        # find the smallest object
        if area < min_area:
            min_area = area
            smallest = co

# create a new copy of resized image to draw the largest and smallest object
img_size = np.copy(img_bright)

# find the center of mass for object
L = cv2.moments(largest)
S = cv2.moments(smallest)
cxl = int(L['m10'] / L['m00'])
cyl = int(L['m01'] / L['m00'])
cxs = int(S['m10'] / S['m00'])
cys = int(S['m01'] / S['m00'])

# Adding text to largest and smallest object
cv2.putText(img_size, "largest", (cxl, cyl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(img_size, "smallest", (cxs, cys), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# draw the contour for largest object and smallest object
cv2.drawContours(img_size, [largest], -1, (0, 255, 0), 2)
cv2.drawContours(img_size, [smallest], -1, (0, 255, 0), 2)

cylinder = 0
triangle = 0
rectangle = 0
circle = 0
for i in contour:
    if i ==2:
        cylinder +=1
    elif i == 3:
        triangle += 1
    elif i == 4:
        rectangle += 1
    else:
        circle += 1

cv2.imshow("Different Shape", img_shape)
cv2.imshow("Check Size", img_size)
cv2.imshow("Detect Object", img_counting)

################################################ find shape,total and size End ################################

################################################ OBJECT COLOR START ################################
# create a new copy of resized image to count the total number of object in different colour.
img_color = np.copy(img_bright)
# Convert to HSV color space
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

# Define the range of color in HSV
lower_red = np.array([0, 255, 255])
upper_red = np.array([10, 255, 255])

lower_green = np.array([60, 50, 50])
upper_green = np.array([80, 255, 255])

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_brown = np.array([5, 10, 100])
upper_brown = np.array([40, 210, 200])

lower_silver = np.array([0, 0, 180])
upper_silver = np.array([180, 50, 255])

# Threshold the image to get a binary mask of the colour pixels
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_green, upper_green)
mask3 = cv2.inRange(hsv, lower_blue, upper_blue)
mask4 = cv2.inRange(hsv, lower_brown, upper_brown)
mask5 = cv2.inRange(hsv, lower_silver, upper_silver)

# Apply a morphological closing operation to merge the nearby same colour area
kernel = np.ones((21, 21), np.uint8)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)
mask4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel)
mask5 = cv2.morphologyEx(mask5, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours4, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours5, _ = cv2.findContours(mask5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check color on the image
def checkColor(contour):
    color = 0
    for co in contour:
        area = cv2.contourArea(co)
        if area > 400:
            color += 1
    return color

# check object with color Red, green, and blue colour
red = checkColor(contours1)
green = checkColor(contours2)
blue = checkColor(contours3)
brown = checkColor(contours4)
silver = checkColor(contours5)

################################################ OBJECT COLOR END ################################

# Display result
print("Number of objects: ", cnt)
print("-----------------------------------------------------------")
print("Area of Largest Object: ", cv2.contourArea(largest))
print("Area of Smallest Object: ", cv2.contourArea(smallest))
print("-----------------------------------------------------------")
print("Number of cylinder: ", cylinder)
print("Number of triangles: ", triangle)
print("Number of rectangles: ", rectangle)
print("Number of circles: ", circle)
print("-----------------------------------------------------------")
print("Number of red objects:" + str(red))
print("Number of green objects:" + str(green))
print("Number of blue objects:" + str(blue))
print("Number of brown objects:" + str(brown))
print("Number of silver objects:" + str(silver))
print("Number of other color objects:" + str(cnt-blue-red-green-brown-silver))
print("-----------------------------------------------------------")

################################################ Extract One By One Start ################################

# Contouring
cont, hier = cv2.findContours(mask, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

result = []

# create a new copy of resized image to extract object one by one
img_obj = np.copy(img_bright)

#draw contour one by one in new img_obj
for co in cont:
    area = cv2.contourArea(co)
    if area > 100:
        black_mask = np.zeros(img_obj.shape[:2], np.uint8)
        cv2.drawContours(black_mask, [co], -1, 255, -1)
        result.append(cv2.bitwise_and(img_obj, img_obj, mask=black_mask))

i = 0
while i < len(result):
    # Display each object in array result one by one
    cv2.imshow("object" + str(i + 1), result[i])
    i = i + 1

################################################ One By One End ################################

################################################ Change BackGround Start ################################

# Create a blue background
blue_bgnd = np.zeros_like(img_bright)
blue_bgnd[:, :, 0] = 255  # blue color
# Copy each object in the image to blue background
cv2.copyTo(img_bright, mask, blue_bgnd)

# Import a background image
table_bgnd = cv2.imread("background.jpg")
height, width = img_bright.shape[:2]
table_bgnd = cv2.resize(table_bgnd, (width, height))
# Copy each object in the image to background image
cv2.copyTo(img_bright, mask, table_bgnd)

cv2.imshow("Blue Background", blue_bgnd)
cv2.imshow("Table Background", table_bgnd)

################################################ Change BackGround End ################################

# cv2.imshow("GreyScale", grey)

cv2.waitKey(0)
