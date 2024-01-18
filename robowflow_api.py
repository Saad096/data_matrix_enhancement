from roboflow import Roboflow
import cv2
import numpy as np
rf = Roboflow(api_key="gEhnvEfmjEMDgnRqLBWl")
project = rf.workspace().project("dm-detection")
model = project.version(1).model
image_size = cv2.imread("image (3).png").shape
background = np.ones(image_size, dtype=np.uint8) * 255
# infer on a local image
dasta_matrix = model.predict("image (3).png", confidence=0, overlap=1).json()['predictions']
for module in dasta_matrix:
    x = int(module['x'])
    y = int(module['y'])
    width = int(module['width'])
    height = int(module['height'])
    cv2.rectangle(background, (x, y), (x + width, y + height), (0, 0, 0), -1)

cv2.imwrite('result_image.png', background)
cv2.imshow('Result Image', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
############################################3333

# from roboflow import Roboflow
# import cv2
# import numpy as np

# # Roboflow setup
# rf = Roboflow(api_key="gEhnvEfmjEMDgnRqLBWl")
# project = rf.workspace().project("dm-detection")
# model = project.version(1).model

# # Load image and detect modules
# image = cv2.imread("image (3).png")
# image_size = image.shape
# background = np.ones(image_size, dtype=np.uint8) * 255
# dasta_matrix = model.predict("image (3).png", confidence=0, overlap=1).json()['predictions']
# modules = [(int(module['x']), int(module['y']), int(module['width']), int(module['height'])) for module in dasta_matrix]

# # Function to join nearest modules
# def join_nearest_modules(modules, distance_threshold=5):
#     joined_modules = []
#     for i in range(len(modules)):
#         if modules[i] not in joined_modules:
#             x1, y1, w1, h1 = modules[i]
#             for j in range(i + 1, len(modules)):
#                 x2, y2, w2, h2 = modules[j]
#                 distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#                 if distance <= distance_threshold:
#                     joined_modules.append((min(x1, x2), min(y1, y2), max(x1+w1, x2+w2) - min(x1, x2), max(y1+h1, y2+h2) - min(y1, y2)))
#     return joined_modules

# # Function to create L-shape from modules (assuming modules are already joined)
# def create_l_shape(modules):
#     # Simple implementation: Select the first two modules as the L-shape
#     return modules[:2]

# # Apply module joining and L-shape formation
# joined_modules = join_nearest_modules(modules)
# l_shape = create_l_shape(joined_modules)

# # Draw joined modules and L-shape on a new image
# valid_datamatrix = np.ones(image_size, dtype=np.uint8)
# for x, y, w, h in joined_modules:
#     cv2.rectangle(valid_datamatrix, (x, y), (x + w, y + h), 255, -1)
# for x, y, w, h in l_shape:
#     cv2.rectangle(valid_datamatrix, (x, y), (x + w, y + h), 128, -1)  # Mark L-shape with a different color

# cv2.imwrite('valid_datamatrix.png', valid_datamatrix)