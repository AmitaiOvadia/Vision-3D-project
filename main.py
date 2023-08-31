# This is a sample Python script.
import numpy as np
import os  # For creating directories and paths
import cv2 # For video processing
import matplotlib.pyplot as plt # For plotting and displaying images
import skimage.io
import skimage.measure
from skimage import io, color, transform, util
from skimage.morphology import binary_closing, disk, binary_erosion
import scipy.stats as stats
import skimage.draw as draw

# load all frames from directory to numpy array


def load_frames_to_array(dir_path):
    file_list = os.listdir(dir_path)
    file_list.sort()

    frame_list = []
    # Loop through the files in the directory
    for i, file in enumerate(file_list):
        # Construct the full file path by joining the directory and file names
        file_path = os.path.join(dir_path, file)
        # Read the image from the file using cv2.imread
        frame = cv2.imread(file_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame_list.append(frame)

    # Convert the frame list to a numpy array using np.array
    frame_array = np.array(frame_list)
    return frame_array


def draw_circle(image, center, radius):
    rr, cc, val = draw.circle_perimeter_aa(int(center[0]), int(center[1]),
                                           int(radius))  # generate circle perimeter coordinates
    image[rr, cc] = val * 255  # draw the circle on the image
    skimage.io.imshow(image)
    skimage.io.show()


def find_ball(img, background):
    img = img - background
    img[img > 170] = 0
    img[img < 20] = 0
    img[img > 0] = 1

    # Show the original image and the mask
    skimage.io.imshow(img)
    skimage.io.show()

    # Apply label()
    labels = skimage.measure.label(img)
    # Get the regionprops
    regions = skimage.measure.regionprops(labels)
    # Get the largest region
    max_region = regions[0]
    for region in regions[1:]:
        if region.area > max_region.area:
            max_region = region

    # Create a mask with the largest region
    mask = np.zeros_like(labels)
    mask[labels == max_region.label] = 255

    # Show the original image and the mask
    skimage.io.imshow(mask)
    skimage.io.show()

    radius = 30
    footprint = disk(radius)  # create a circular footprint
    closed = binary_closing(mask, footprint)  # get the closed image
    # skimage.io.imshow(closed)
    # skimage.io.show()

    radius = 1
    footprint = disk(radius)  # create a circular footprint
    eroded = binary_erosion(closed, footprint)
    edge = closed.astype(int) - eroded.astype(int)
    # # Show the result
    # io.imshow(edge)
    # io.show()
    rows, cols = np.where(edge == 1)  # get the row and column indices of pixels equal 1
    points = np.array(list(zip(rows, cols)))   # zip them together to get a list of coordinates

    # num_points = 100
    num_points = points.shape[0]
    indices = np.random.choice(points.shape[0], num_points, replace=False)  # select 50 random indices
    furthest_points = []  # a list to store the furthest points
    point_pairs = []
    points_array = np.zeros((num_points, 3, 2))
    for i, ind in enumerate(indices) :  # loop over the selected indices
        point = points[ind]  # get the current point
        distances = np.linalg.norm(points - point,
                                   axis=1)  # compute the L2 norm between the current point and all other points
        furthest_index = np.argmax(distances)  # find the index of the point with the maximum distance
        furthest_point = points[furthest_index]  # get the furthest point
        furthest_points.append(furthest_point)  # append it to the list
        furthest_distance = distances[furthest_index]
        points_array[i, 0, :] = point
        points_array[i, 1, :] = furthest_point
        points_array[i, 2, 1] = furthest_distance
        point_pairs.append((point, furthest_point, furthest_distance))

    distances = points_array[:, 2, 1]
    mad = stats.median_abs_deviation(distances)  # compute the MAD along the axis
    median = np.median(distances)  # compute the median along the axis
    threshold = 1  # set a threshold for outliers
    outliers = np.abs(distances - median) / mad > threshold  # create a boolean mask for outliers
    final_pnts = points_array[~outliers, :, :]
    final_pnts = np.concatenate((final_pnts[:, 0, :], final_pnts[:, 1, :]), axis=0)
    final_radius = int(median / 2)
    final_center = np.round(np.mean(final_pnts, axis=0)).astype(int)
    return final_radius, final_center


if __name__ == '__main__':
    frame_array = load_frames_to_array("ball frames")
    background = frame_array[0]
    first = frame_array[3]
    frame = first
    num_frames = frame_array.shape[0] - 1
    for frame_num in range(20, num_frames):
        frame = frame_array[frame_num]
        final_radius, final_center = find_ball(frame, background)
        draw_circle(frame, final_center, final_radius)

