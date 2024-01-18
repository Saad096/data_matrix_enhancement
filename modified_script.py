import cv2
import numpy as np

# Load the image
image = cv2.imread('image (2).png')  # Replace 'path_to_your_image.jpg' with the actual file path

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur for smoothing
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)e

# Draw contours on the original image
contour_img = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# Display the result
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#########################################################################################################
# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread("image (2).png")  # Replace with your image path

# # Preprocessing
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

# # Contour detection
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Generate data matrix
# data_matrix = []
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)  # Approximate contours
#     x, y, w, h = cv2.boundingRect(approx)  # Get bounding rectangles
#     data_matrix.append([x, y, w, h])  # Store coordinates and dimensions

# # Print or visualize the data matrix
# print(data_matrix)  # Plain text output
# # Or, for visualization:
# for row in data_matrix:
#     x, y, w, h = row
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangles on original image
# cv2.imshow("Image with Contours", image)
# cv2.waitKey(0)

#########################################################################################################

# """ Author: BediS
# Generated on: December 9th, 2023"""

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.transforms import Affine2D
# import mpl_toolkits.axisartist.floating_axes as floating_axes
# import cv2

# # Read the Image
# image = cv2.imread("3 (1).png")
# print(f'Image Size: {image.size}')

# # Convert the image to Grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)

# gray_blur = cv2.medianBlur(gray, 7)  # Only Odd Integer Number

# # Apply thresholding to create a binary image
# # Apply thresholding to create a binary image
# _, thresh = cv2.threshold(gray_blur, 215, 300, cv2.THRESH_BINARY)

# # Display the binary image after thresholding
# plt.subplot(1, 3, 1)
# plt.imshow(thresh, cmap='gray')
# plt.title("Binary Image")

# # Find Contours in the binary image
# contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # Display the image with detected contours
# contour_image = np.zeros_like(image)
# cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
# plt.subplot(1, 3, 2)
# plt.imshow(contour_image)
# plt.title("Contours")

# # Extract x and y coordinates of the contours (dots)
# points = []
# x = []
# y = []
# for contour in contours:
#     # Calculate the moments of the contour
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         # Calculate centroid coordinates (x, y)
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         points.append((cX, cY))
#         x.append(cX)
#         y.append(cY)

# # Display the image with detected points
# plt.subplot(1, 3, 3)
# plt.imshow(image)
# plt.plot(np.array(x), np.array(y), 'ro')
# plt.title("Detected Points")

# plt.show()

# #  Calculate the average distances between consecutive points in x and y direction
# # x_dist = [points[i+1][0] - points[i][0] for i in range(len(points)-1)]
# # y_dist = [points[i+1][1] - points[i][1] for i in range(len(points)-1)]
# # print(f'Average Distances between X points: {x_dist}')
# # print(f'Average Distances between Y points: {y_dist}')

# #  Calculate the grid size (average distances)
# # grid_size_x = round(sum(x_dist) / len(x_dist))
# # grid_size_y = round(sum(y_dist) / len(y_dist))
# # print(f'Grid Size [X]: {grid_size_x}')
# # print(f'Grid Size [Y]: {grid_size_y}')

# if len(points) > 1:
#     x_dist = [points[i + 1][0] - points[i][0] for i in range(len(points) - 1)]
#     y_dist = [points[i + 1][1] - points[i][1] for i in range(len(points) - 1)]
    
#     print(f'Average Distances between X points: {x_dist}')
#     print(f'Average Distances between Y points: {y_dist}')

#     # Calculate the grid size (average distances)
#     grid_size_x = round(sum(x_dist) / len(x_dist))
#     grid_size_y = round(sum(y_dist) / len(y_dist))
#     print(f'Grid Size [X]: {grid_size_x}')
#     print(f'Grid Size [Y]: {grid_size_y}')
# else:
#     print("Not enough points to calculate distances.")

# #  Compare the Image and 2D Bar Code
# plt.subplot(1, 2, 1)
# plt.imshow(gray)
# plt.title("Image")
# plt.subplot(1, 2, 2)
# plt.plot(np.array(x), np.array(y), 'ko')
# plt.title("2D Bar Code")
# plt.gca().set_aspect('equal')
# plt.gca().invert_yaxis()
# plt.show()







# # #  Plot the 2D Bar Code
# # plt.plot(np.array(x), np.array(y), 'ko')
# # plt.title("2D Bar Code")
# # plt.gca().set_aspect('equal')
# # plt.gca().invert_yaxis()
# # plt.show()

# # # Plotting the dots in the image
# # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# # #plt.scatter(x, y, color='red', s=5)
# # plt.title('Detected dots')
# # plt.show()

# # plt.style.use('dark_background')
# # plt.plot(np.array(x), np.array(y), 'wo')


# import cv2
# import numpy as np

# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Apply Gaussian blur for noise reduction
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)

#     # Apply adaptive thresholding for creating a binary image
#     _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     return thresholded

# def enhance_contrast(image):
#     # Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(image)

#     return enhanced

# def main():
#     # Replace 'your_image_path.png' with the path to your Data Matrix image
#     image_path = 'image (2).png'

#     # Preprocess the image
#     preprocessed_image = preprocess_image(image_path)

#     # Enhance contrast
#     contrast_enhanced_image = enhance_contrast(preprocessed_image)

#     # Display the original and processed images (optional)
#     cv2.imshow('Original Image', cv2.imread(image_path))
#     cv2.imshow('Processed Image', np.hstack([preprocessed_image, contrast_enhanced_image]))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




