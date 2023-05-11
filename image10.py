import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image file
img = cv2.imread("image10.jpg")
# Reduce size of output to 10% with preserve aspect ratio
img_resized = cv2.resize(img, None, fx=0.1, fy=0.1)

# Create a copy of the image
img_copy = np.copy(img_resized)
# Convert to RGB to display via matplotlib
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
# Find the coordinates of 4 corner points for transformation matrix
plt.imshow(img_copy)
plt.show()

# All points in format [x, y]
pt_A = [33, 360]
pt_B = [276, 356]
pt_C = [227, 45]
pt_D = [69, 35]

# Calculate the width of tissue box
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

# Calculate the height of tissue box
height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

# Print the height and width obtained
print("Width of AD: ", width_AD)
print("Width of BC: ", width_BC)
print("Max Width: ",maxWidth)
print("Height of AB: ", height_AB)
print("Height of CD: ", height_CD)
print("Max Height: ", maxHeight)

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

# Calculate the perspective transformation matrix (M)
M = cv2.getPerspectiveTransform(input_pts,output_pts)

# Apply the perspective transformation to an image
img_transformed = cv2.warpPerspective(img_resized,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

# Brighten the image
brightness = 1.5
img_bright = cv2.convertScaleAbs(img_transformed, alpha=brightness, beta=0)
cv2.imshow("Bright Image ",img_bright)

cv2.imshow("Original Image", img_resized)
cv2.imshow("Transformed Image", img_transformed)
cv2.waitKey(0)
