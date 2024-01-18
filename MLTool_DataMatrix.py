""" Author: BediS
Generated on: January 12th, 2024"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import io
import ast

# Read the MLTool extracted Data
# ml_data = open("MLTool_Data/DM4.txt", 'r')
# print(f'MLTool_Data: {ml_data.read()}')
# print(type(ml_data))

with open("MLTool_Data/DM8.txt", 'r') as file:
    file_data = file.read()
    list_of_dict = ast.literal_eval(file_data)
# print(list_of_dict)

x_data = []
y_data = []
for i, dictionary in enumerate(list_of_dict, start=1):
    globals()[f'dic{i}'] = dictionary
    print(f'Dictionary{i}:', dictionary)
    keys = list(dictionary.keys())
    print(keys)
    values = list(dictionary.values())
    print(values)
    if values[6] == 1:   ## Set either individually 0,1,2,3... or "0 or 1 or 2 or 3" depending on the class_id of data
        x_data.append(values[0])
        y_data.append(values[1])
print(f'X-Coordinates: {x_data}')
print(f'No. of X-Coordinates: {len(x_data)}')
print(f'Y-Coordinates: {y_data}')
print(f'No. of Y-Coordinates: {len(y_data)}')

# Calculate Coordinates of Modules
points = list(zip(x_data, y_data))
print(f'Coordinates of Modules: {points}')

#  Calculate the average distances between consecutive points in x and y direction
x_dist = [x_data[i+1] - x_data[i] for i in range(len(x_data)-1)]
y_dist = [y_data[i+1] - y_data[i] for i in range(len(y_data)-1)]
print(f'Average Distances between X points: {x_dist}')
print(f'Average Distances between Y points: {y_dist}')


# Generate squares around points
def plot_squares(coordinates, side_length):
    for coord in coordinates:
        x_data, y_data = coord
        square = plt.Rectangle((x_data-side_length/2, y_data-side_length/2), side_length, side_length, edgecolor='black', facecolor='black')
        plt.gca().add_patch(square)


# Set an optimal range for and calculate side length
# side_length = abs((12))

# limit = range(12, 17)
# for i in x_dist:
#     if i in limit:
#         side_length = abs(i)

limit = range(9, 10)
for i in x_dist:
    if i in limit:
        side_length = abs(np.median(limit))
print(f'Side Length: {side_length}')


# Plotting squares around the coordinates
plt.scatter(x_data, y_data, color='blue', marker='s', label='Points')
plot_squares(points, side_length)
plt.gca().invert_yaxis()
plt.savefig("Output_2D_DataMatrix/DM8.png")
plt.show()