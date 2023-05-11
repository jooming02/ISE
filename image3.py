import cv2
import numpy as np

#denoise
img = cv2.imread("image3.jpg")
img_copy = img
color_img = img
img_obj = img
img_bg = img

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey = cv2.medianBlur(grey, 21)
grey = cv2.equalizeHist(grey)
grey = cv2.equalizeHist(grey)
r, c = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY_INV)
msk = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))


c = cv2.morphologyEx(c, cv2.MORPH_DILATE, kernel=ker, iterations=19)

c = cv2.resize(c, None, fx=0.1, fy=0.1)
img = cv2.resize(img, None, fx=0.1, fy=0.1)
img_copy = cv2.resize(img_copy, None, fx=0.1, fy=0.1)
color_img = cv2.resize(color_img, None, fx=0.1, fy=0.1)

N, idx, stats, cent = cv2.connectedComponentsWithStats(c)

cnt = 0


for s in stats:
    x = s[0]
    y = s[1]
    w = s[2]
    h = s[3]
    
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    # counter += 1
    # print(w)
    
    if w > 13 and w < 70:
        cnt += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        img_copy[y:y+h, x:x+w, :] = 255

img_obj2 = img_copy


#img_copy
grey2 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_cont = np.copy(grey2)
# grey2 = cv2.medianBlur(grey2, 13)
# grey2 = cv2.equalizeHist(grey2)
# grey2 = cv2.equalizeHist(grey2)
r, c2 = cv2.threshold(grey2, 140, 255, cv2.THRESH_BINARY_INV)
c2 = cv2.GaussianBlur(c2, (5, 5), 3)
# ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


# c2 = cv2.morphologyEx(c2, cv2.MORPH_CLOSE, kernel=ker, iterations=10)

N, idx, stats, cent = cv2.connectedComponentsWithStats(c2)

for s in stats:
    x = s[0]
    y = s[1]
    w = s[2]
    h = s[3]
    print("width", w)
    
    if w > 100 and w < 400:
        
        cnt += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    if w == 136:    
        img_copy[y:y+h, x:x+w, :] = 255

img_obj3 = img_copy

#contouring(first image)
cont,hier = cv2.findContours(c,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
contour = []
for co in cont:
    print("Area : ", cv2.contourArea(co), ", ---- Perimeter : ",cv2.arcLength(co,True))
    area = cv2.contourArea(co)
    eps=0.07*cv2.arcLength(co,True)
    approx_cont = cv2.approxPolyDP(co,eps,True)
    if area >250 and area < 5000:
        contour.append(len(approx_cont))   
        cv2.drawContours(img,approx_cont,-1,(255,0,0),5)

triangle = 0
rectangle = 0
circle = 0
cylinder = 0
for i in contour:
    if i == 2:
        cylinder += 1
    elif i == 3:
        triangle += 1
    elif i == 4:
        rectangle += 1
    elif i > 4:
        circle += 1

#contouring (img copy) find circle
grey2 = cv2.medianBlur(img_cont, 13)
grey2 = cv2.equalizeHist(grey2)
grey2 = cv2.equalizeHist(grey2)
r, c2 = cv2.threshold(grey2, 70, 255, cv2.THRESH_BINARY_INV)
c2 = cv2.GaussianBlur(c2, (5, 5), 3)
# ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


# c2 = cv2.morphologyEx(c2, cv2.MORPH_CLOSE, kernel=ker, iterations=10)

cont2,hier2 = cv2.findContours(c2,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
contour = []
for co in cont2:
    print("Area : ", cv2.contourArea(co), ", ---- Perimeter : ",cv2.arcLength(co,True))
    area = cv2.contourArea(co)
    eps=0.01*cv2.arcLength(co,True)
    approx_cont = cv2.approxPolyDP(co,eps,True)
    if area >12000:
        contour.append(len(approx_cont))   
        cv2.drawContours(img,[approx_cont],-1,(255,0,0),5)

for i in contour:
    if i == 3:
        triangle += 1
    elif i == 4:
        rectangle += 1
    else:
        circle += 1

print("circle", circle)

#find tissue
grey3 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# control = 0.1 # Change this value to control the amount of brightness reduction
# grey3 = control * np.log(1 + grey3)
# grey3 = cv2.medianBlur(grey3, 21)
# grey3 = cv2.equalizeHist(grey3)
# grey3 = cv2.equalizeHist(grey3)
r, c3 = cv2.threshold(grey3, 150, 255, cv2.THRESH_BINARY_INV)
ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


c3 = cv2.morphologyEx(c3, cv2.MORPH_CLOSE, kernel=ker, iterations=10)

cont2,hier2 = cv2.findContours(c3,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
contour = []
for co in cont2:
    print("Area : ", cv2.contourArea(co), ", ---- Perimeter : ",cv2.arcLength(co,True))
    area = cv2.contourArea(co)
    eps=0.08*cv2.arcLength(co,True)
    approx_cont = cv2.approxPolyDP(co,eps,True)
    if area >12000:
        contour.append(len(approx_cont))   
        cv2.drawContours(img,approx_cont,-1,(255,0,0),5)

for i in contour:
    if i == 3:
        triangle += 1
    elif i == 4:
        rectangle += 1
    else:
        circle += 1


################################################ OBJECT COLOR START ################################
# create a new copy of resized image to count the total number of object in different colour.
img_color = np.copy(img)
# img_color = cv2.GaussianBlur(img_color, (5, 5), 3)
# Convert to HSV color space
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

# Define the range of color in HSV
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 50, 50])

brown_l = np.array([5, 10, 100]) 
brown_u = np.array([40, 210, 200])

# brown_l = np.array([10, 30, 150]) # Hue: 10-20, Saturation: 30-255, Value: 150-255
# brown_u = np.array([20, 120, 160])

# Threshold the image to get a binary mask of the colour pixels
mask1 = cv2.inRange(hsv, lower_black, upper_black)
mask2 = cv2.inRange(hsv, brown_l, brown_u)

#denoise
mask2 = cv2.medianBlur(mask2, 27)


# Apply a morphological closing operation to merge the nearby same colour area
kernel = np.ones((7, 7), np.uint8)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=3)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=5)


# Find contours in the mask
contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Check color on the image
def checkColor(contour):
    color = 0
    for co in contour:
        area = cv2.contourArea(co)
        eps=0.04*cv2.arcLength(co,True)
        approx_cont = cv2.approxPolyDP(co,eps,True)
        print(area)
        if area >500 and area < 12000:
            cv2.drawContours(color_img,[approx_cont],-1,(255,0,0),5)
            color += 1
    return color


# check object with color Red, green, and blue colour
black = checkColor(contours1)
brown = checkColor(contours2)


################################################ OBJECT COLOR END ################################

################################################ Extract One By One Start ################################

# Contouring
cont, hier = cv2.findContours(c, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)
cont2, hier = cv2.findContours(c2, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)
cont3, hier = cv2.findContours(c3, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

result = []

# create a new copy of resized image to extract object one by one
img_obj = cv2.resize(img_obj, None, fx=0.1, fy=0.1)

#draw contour one by one in new img_obj
for co in cont:
    area = cv2.contourArea(co)
    if area >250 and area < 5000:
        mask = np.zeros(img_obj.shape[:2], np.uint8)
        cv2.drawContours(mask, [co], -1, 255, -1)
        result.append(cv2.bitwise_and(img_obj, img_obj, mask=mask))

for co in cont2:
    area = cv2.contourArea(co)
    if area >12000:
        mask = np.zeros(img_obj.shape[:2], np.uint8)
        cv2.drawContours(mask, [co], -1, 255, -1)
        result.append(cv2.bitwise_and(img_obj, img_obj, mask=mask))

for co in cont3:
    area = cv2.contourArea(co)
    if area >12000:
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
mask = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel=morp_ker, iterations=1)
#cv2.imshow("Morphology Close", mask)

# Fill the region with white color
cv2.floodFill(mask, None, (204, 80), (255, 0, 0))
cv2.floodFill(mask, None, (204, 40), (255, 0, 0))
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


print("Number of black objects:" + str(black))
print("Number of brown objects:" + str(brown))
print("Number of other color objects:" + str(cnt-black-brown))
print(cnt)
cv2.imshow("c", c)  
# cv2.imshow("ori", img)
# cv2.imshow("img_copy", img_copy)
# cv2.imshow("c2", c2)
# cv2.imshow("color_img", color_img) 
# cv2.imshow("c3", c3)
# cv2.imshow("m2", mask2)
cv2.waitKey(0)