import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image5.jpg")
# Reduce size of output to 10% with preserve aspect ratio
img_resized = cv2.resize(img, None, fx=0.1, fy=0.1)

# Create a copy of the image
img_copy = np.copy(img_resized)

# Convert to RGB to display via matplotlib
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
# Find the coordinates of 4 points for transformation matrix
plt.imshow(img_copy)
plt.show()

# All points in format [cols, rows]
pt_A = [33, 360]
pt_B = [276, 356]
pt_C = [227, 45]
pt_D = [69, 35]


width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)

out = cv2.warpPerspective(img_resized,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

cv2.imshow("Original Image", img_resized)
cv2.imshow("Res", out)
cv2.waitKey(0)