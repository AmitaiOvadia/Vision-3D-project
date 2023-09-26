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
from skimage.transform import hough_circle, hough_circle_peaks
from PIL import Image
from skimage import data, io, measure, draw

APPER_THRESHOLD = 190
LOWER_THRESHOLD = 20


def show_image(image):
    io.imshow(image)
    io.show()



class BallDitector2D:
    def __init__(self, ball_frames_dir):
        self.ball_frames_dir = ball_frames_dir
        self.upper_threshold = APPER_THRESHOLD
        self.background = self.get_background_image()
        self.cur_frame = 1
        self.centers = None
        self.radii = None

    def find_edge_points(self, mask):
        edge = BallDitector2D.find_edges(mask)
        # center, radious = self.find_circle_hough(edge)
        rows, cols = np.where(edge == 1)  # get the row and column indices of pixels equal 1
        points = np.array(list(zip(rows, cols)))  # zip them together to get a list of coordinates
        median = np.median(points, axis=0)
        distace_from_median = np.linalg.norm(points - median, axis=1)
        points = points[distace_from_median < 200]
        return points

    def fit_circle_to_edge(self, points):
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

        final_pnts, median = self.remove_outliers(points_array)
        final_radius = median / 2
        final_center = np.mean(final_pnts, axis=0)
        return final_center, final_radius

    def remove_outliers(self, points_array):
        distances = points_array[:, 2, 1]
        mad = stats.median_abs_deviation(distances)  # compute the MAD = median(|Yi – median(Yi|) along the axis
        median = np.median(distances)  # compute the median along the axis
        threshold = 1  # set a threshold for outliers
        outliers = np.abs(distances - median) / mad > threshold  # create a boolean mask for outliers
        points_array = points_array[~outliers, :, :]
        final_pnts = np.concatenate((points_array[:, 0, :], points_array[:, 1, :]), axis=0)
        return final_pnts, median

    def find_ball_in_frame(self, img):
        mask = self.segment_ball(img)
        if self.cur_frame < 90:
            points = self.find_edge_points(mask)
            final_center, final_radius = self.fit_circle_to_edge(points)
        else:
            self.upper_threshold = 170
            edge = BallDitector2D.find_edges(mask)
            final_center, final_radius = self.find_circle_hough(edge)
        return final_radius, final_center

    def segment_ball(self, img):
        img = img - self.background
        # upper_threshold = APPER_THRESHOLD if self.cur_frame not in GROUND_FRAMES_T2_sk2 else 130
        img[img > self.upper_threshold] = 0
        img[img < LOWER_THRESHOLD] = 0
        img[img > 0] = 1
        footprint = disk(3)  # create a circular footprint
        mask = binary_opening(img, footprint)
        footprint = disk(5)  # create a circular footprint
        mask = binary_closing(mask, footprint)  # get the closed image

        min_size = 200  # Define the minimum size of the objects to keep
        mask = remove_small_objects(mask, min_size=min_size, connectivity=1)  # Remove small objects
        return mask

    def draw_circle(self, image, center, radius, save=False, do_draw=True):
        image = Image.fromarray(image)
        image = image.convert('RGB')
        rr1, cc1, val1 = draw.circle_perimeter_aa(int(center[0]), int(center[1]),
                                               int(radius))  # generate circle perimeter coordinates
        rr2, cc2, val2 = draw.circle_perimeter_aa(int(center[0]), int(center[1]),
                                                  int(radius + 1))  # generate circle perimeter coordinates
        rr3, cc3, val3 = draw.circle_perimeter_aa(int(center[0]), int(center[1]),
                                                  int(radius - 1))  # generate circle perimeter coordinates
        rr4, cc4, val4 = draw.circle_perimeter_aa(int(center[0]), int(center[1]),
                                                  int(radius + 2))  # generate circle perimeter coordinates
        image = np.array(image)
        if np.max(image) <= 1:
            image = image*255
        image[rr1, cc1, 0] = val1 * 255  # draw the circle on the image
        image[rr2, cc2, 0] = val2 * 255  # draw the circle on the image
        image[rr3, cc3, 0] = val3 * 255  # draw the circle on the image
        image[rr4, cc4, 0] = val4 * 255  # draw the circle on the image
        if do_draw:
            show_image(image)
        if save:
            io.imsave(f'validation throw/throw 1 circles/image{self.cur_frame}.png', image)

    def find_ball_all_frames(self):
        file_list = os.listdir(self.ball_frames_dir)
        # file_list.sort()
        radii = []
        centers = []
        for i, file in enumerate(file_list):
            if i < 1: continue
            self.cur_frame = i
            if i == 28:
                pass
            print(i)
            file_path = os.path.join(self.ball_frames_dir, file)
            frame = BallDitector2D.read_frame(file_path)
            radius, center = self.find_ball_in_frame(frame)
            if i % 1 == 0:
                self.draw_circle(frame, center, radius, save=True, do_draw=False)
            radii.append(radius)
            centers.append(center)
        self.radii = np.array(radii)
        self.centers = np.array(centers)
        self.save_results()
        return radii, centers

    def get_background_image(self):
        file_list = os.listdir(self.ball_frames_dir)
        background = self.read_frame(os.path.join(self.ball_frames_dir, file_list[0]))
        return background

    def save_results(self):
        np.save('validation throw\centers_throw_1_skip_1.npy', self.centers)
        np.save(r'validation throw\radii_throw_1_skip_1.npy', self.radii)

    def plot_ball_trajectory_3D(self, load=False):
        plt.imshow(self.background)
        trajectory = self.centers.astype(int)
        # Plot the trajectory points
        plt.plot(trajectory[:, 1], trajectory[:, 0], marker='o', color='red', markersize=5)
        plt.axis('off')  # Turn off axis
        plt.show()

    def find_circle_hough(self, edge_img):
        props = measure.regionprops(edge_img)[0]  # assuming there is only one blob
        min_y, min_x, max_y, max_x = props.bbox  # get the bounding box coordinates
        avv = int((max_x - min_x + max_y - min_y)/4)
        hough_radii = np.arange(avv - int(avv/2), avv + int(avv/2))
        hough_res = hough_circle(edge_img, hough_radii)

        # Select the most prominent circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=1)
        # self.draw_circle(edge_img, (cy, cx), radii, save=False, do_draw=True)
        return (cy, cx), radii


    @staticmethod
    def find_edges(binary_image, radius=2):
        footprint = disk(radius)  # create a circular footprint
        eroded = binary_erosion(binary_image, footprint)
        edge = binary_image.astype(int) - eroded.astype(int)
        return edge

    @staticmethod
    def read_frame(frame_path):
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


if __name__ == '__main__':
    path = "validation throw/throw 1"
    ball_detector = BallDitector2D(path)

    radii, centers = ball_detector.find_ball_all_frames()
    background = ball_detector.get_background_image()
    # centers = np.load('Ball videos 4\centers_throw_2_skip_2.npy')
    # radii = np.load(r'Ball videos 4\radii_throw_2_skip_2.npy')
    # centers = array_2_list_of_tuples(centers)
    ball_detector.plot_ball_trajectory_3D(load=False)
    a=0