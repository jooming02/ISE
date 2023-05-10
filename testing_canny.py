import cv2
import numpy as np

def f(x):
    # print the control value
    print(x)
    return(x)
#ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
img = cv2.imread("image12.jpg")

print("Size of image: ", img.shape)

# Reduce size of output to 10% with preserve aspect ratio
scale_pct = 10
width = int(img.shape[1] * scale_pct / 100)
height = int(img.shape[0] * scale_pct / 100)
size = (width, height)

rsz_img = cv2.resize(img, size)
print("Size of image: ", rsz_img.shape)

# Convert to grayscale image
gs = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY)

# Convert to binary color
r,bw  = cv2.threshold(gs,200,255,cv2.THRESH_BINARY_INV)
img_binary = cv2.adaptiveThreshold(gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


# cv2.imshow("Original Image",rsz_img)
# cv2.imshow("GreyScale",gs)
# cv2.imshow("BlackWhite",bw)
# cv2.imshow("xxx",img_binary)

mask = np.zeros_like(rsz_img)

cv2.namedWindow("Control")
cv2.createTrackbar("x","Control",80,500,f)
cv2.createTrackbar("y","Control",170,500,f)


while(True):
    x=cv2.getTrackbarPos("x","Control")
    y=cv2.getTrackbarPos("y","Control")
    edges = cv2.Canny(rsz_img, x, y)
    #res = cv2.morphologyEx(bw,cv2.MORPH_DILATE,ker,iterations=i)
    cv2.imshow("Edge Detected Image", edges)

    cont, hier = cv2.findContours(edges, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)
    cv2.drawContours(mask, cont, -1, (255, 255, 255), 2)
    # Region filling
    #cv2.fillPoly(mask, cont, (255, 255, 255))
    cv2.imshow("Mask", mask)

    key=cv2.waitKey(5)
    if(key==32 or x==-1):
        break;


# Brightness adjustment
# brightness = 2
# img_bright = cv2.convertScaleAbs(rsz_img, alpha=brightness, beta=0)
# cv2.imshow("Brightness",img_bright)
# gs_b = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
# cv2.imshow("GreyScale_2",gs_b)