""" Author: BediS
Generated on: December 9th, 2023"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import cv2

# Read the Image
image = cv2.imread("3 (1).png")
print(f'Image Size: {image.size}')

# Convert the image to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)

gray_blur = cv2.medianBlur(gray, 7)  # Only Odd Integer Number

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray_blur, 215, 300, cv2.THRESH_BINARY)
# thresh = cv2.adaptiveThreshold(gray, 90, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Find Contours in the binary image
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Extract x and y coordinates of the contours (dots)
points = []
x = []
y = []
for contour in contours:
    # Calculate the moments of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        # Calculate centroid coordinates (x, y)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        points.append((cX, cY))
# Show extracted points
for point in points:
    print(f'X: {point[0]}, Y: {point[1]}')
    x.append(point[0])
    y.append(point[1])
print(f'Total No. of Points: {len(points)}')
print(f'Coordinates of Points: {points}')
print(f'X-array of points: {x}')
print(f'Y-array of points: {y}')

#  Plot the Contour/Centroid points on the Image
plt.plot(np.array(x), np.array(y), 'ro')
plt.show()

#  Calculate the average distances between consecutive points in x and y direction
x_dist = [points[i+1][0] - points[i][0] for i in range(len(points)-1)]
y_dist = [points[i+1][1] - points[i][1] for i in range(len(points)-1)]
print(f'Average Distances between X points: {x_dist}')
print(f'Average Distances between Y points: {y_dist}')

#  Calculate the grid size (average distances)
grid_size_x = round(sum(x_dist) / len(x_dist))
grid_size_y = round(sum(y_dist) / len(y_dist))
print(f'Grid Size [X]: {grid_size_x}')
print(f'Grid Size [Y]: {grid_size_y}')

#  Compare the Image and 2D Bar Code
plt.subplot(1, 2, 1)
plt.imshow(gray)
plt.title("Image")
plt.subplot(1, 2, 2)
plt.plot(np.array(x), np.array(y), 'ko')
plt.title("2D Bar Code")
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
plt.show()







# #  Plot the 2D Bar Code
# plt.plot(np.array(x), np.array(y), 'ko')
# plt.title("2D Bar Code")
# plt.gca().set_aspect('equal')
# plt.gca().invert_yaxis()
# plt.show()

# # Plotting the dots in the image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# #plt.scatter(x, y, color='red', s=5)
# plt.title('Detected dots')
# plt.show()

# plt.style.use('dark_background')
# plt.plot(np.array(x), np.array(y), 'wo')






