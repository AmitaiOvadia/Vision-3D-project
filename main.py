# This is a sample Python script.
import numpy as np
import os  # For creating directories and paths
import cv2 # For video processing
import matplotlib.pyplot as plt # For plotting and displaying images
import skimage.io
import skimage.measure
from skimage import io, color, transform, util
from skimage.morphology import binary_closing, disk, binary_erosion, binary_opening, remove_small_objects
import scipy.stats as stats
import skimage.draw as draw
from scipy.ndimage.measurements import center_of_mass
# load all frames from directory to numpy array
from skimage import io, color, transform
from skimage.feature import peak_local_max
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.signal import convolve2d


def draw_circle(image, center, radius):
    rr, cc, val = draw.circle_perimeter_aa(int(center[0]), int(center[1]),
                                           int(radius))  # generate circle perimeter coordinates
    image[rr, cc] = val * 255  # draw the circle on the image
    skimage.io.imshow(image)
    skimage.io.show()


def show_image(image):
    io.imshow(image)
    io.show()


def find_edges(binary_image, radius=2):
    footprint = disk(radius)  # create a circular footprint
    eroded = binary_erosion(binary_image, footprint)
    edge = binary_image.astype(int) - eroded.astype(int)
    return edge


def fit_circle_to_edge(points):
    num_points = points.shape[0]
    indices = np.random.choice(points.shape[0], num_points, replace=False)  # select N random indices
    furthest_points = []  # a list to store the furthest points
    point_pairs = []
    points_array = np.zeros((num_points, 3, 2))
    for i, ind in enumerate(indices):  # loop over the selected indices
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

    final_pnts, median = remove_outliers(points_array)
    final_radius = int(median / 2)
    final_center = np.round(np.mean(final_pnts, axis=0)).astype(int)
    return final_center, final_radius


def remove_outliers(points_array):
    distances = points_array[:, 2, 1]
    mad = stats.median_abs_deviation(distances)  # compute the MAD along the axis
    median = np.median(distances)  # compute the median along the axis
    threshold = 1  # set a threshold for outliers
    outliers = np.abs(distances - median) / mad > threshold  # create a boolean mask for outliers
    points_array = points_array[~outliers, :, :]
    final_pnts = np.concatenate((points_array[:, 0, :], points_array[:, 1, :]), axis=0)
    return final_pnts, median


def find_edge_points(mask):
    edge = find_edges(mask)
    rows, cols = np.where(edge == 1)  # get the row and column indices of pixels equal 1
    points = np.array(list(zip(rows, cols)))  # zip them together to get a list of coordinates
    median = np.median(points, axis=0)
    distace_from_median = np.linalg.norm(points - median, axis=1)
    points = points[distace_from_median < 200]
    return points

# def efficient_binary(mask, footpring, operation):



def segment_ball(img, background):
    img = img - background
    img[img > 170] = 0
    img[img < 20] = 0
    img[img > 0] = 1
    footprint = disk(3)  # create a circular footprint
    mask = binary_opening(img, footprint)
    footprint = disk(5)  # create a circular footprint
    mask = binary_closing(mask, footprint)  # get the closed image
    return mask


def find_ball_in_frame(img, background):
    mask = segment_ball(img, background)
    points = find_edge_points(mask)
    final_center, final_radius = fit_circle_to_edge(points)
    return final_radius, final_center


def read_frame(frame_path):
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def get_background_image(dir_path):
    file_list = os.listdir(dir_path)
    # file_list.sort()
    background = read_frame(os.path.join(dir_path, file_list[0]))


def find_ball_all_frames(dir_path):
    file_list = os.listdir(dir_path)
    # file_list.sort()
    background = read_frame(os.path.join(dir_path, file_list[0]))
    radii = []
    centers = []
    for i, file in enumerate(file_list):
        if i < 1: continue
        # if i > 30: break
        print(i)
        file_path = os.path.join(dir_path, file)
        frame = read_frame(file_path)
        radius, center = find_ball_in_frame(frame, background)
        # draw_circle(frame, center, radius)
        radii.append(radius)
        centers.append(center)

    np.save('centers_video_2.npy', np.array(centers))
    np.save('radii_video_2.npy', np.array(radii))
    return radii, centers, background


def plot_ball_trajectory_3D(background, trajectory):
    plt.imshow(background)
    # Plot the trajectory points
    plt.plot(trajectory[:, 1], trajectory[:, 0], marker='o', color='red')

    plt.axis('off')  # Turn off axis
    plt.show()


if __name__ == '__main__':
    # radii, centers = find_ball_all_frames("ball frames video 2")
    centers = np.load('centers_video_2.npy')
    radii = np.load('radii_video_2.npy')
    centers = np.array(centers)
    radii = np.array(radii)
    a=0

    plot_ball_trajectory_3D(background, centers)

