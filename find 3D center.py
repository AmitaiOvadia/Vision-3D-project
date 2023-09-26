import scipy.io as sio
import numpy as np
# import pygame
import math
import matplotlib.pyplot as plt  # For plotting and displaying images
import pandas as pd
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# one estimation formula can be: distance = size_obj * focal_length / size_obj_on_sensor

MOV3_K_MATRIX = np.array([[1.5873e03, 0, 960.9],
                          [0, 1.5757e3, 561.8347],
                          [0, 0, 1]])

MOV4_K_MATRIX = np.array([[8.38e+02, 0, 956.048],
                          [0, 830.906, 545.236],
                          [0, 0, 1]])

ValMov_K_MATRIX = np.array([[1555.11670496190,	0,	960.377782098074],
                            [0,	1541.73781272614,	565.833760046198],
                            [0,	0,	1]])


MOV3_P_MATRIX = np.array([[1.5873e03, 0, 960.9, 0],
                          [0, 1.5757e3, 561.8347, 0],
                          [0, 0, 1, 0]])

MOV3_PINV_MATRIX1 = np.linalg.pinv(MOV3_P_MATRIX)

MOV3_PINV_MATRIX2 = np.array([[-0.000593920071308024, -4.91334853074069e-05, 0.892119023580987]
                                 , [4.18938354926008e-05, -0.000631928273524856, 0.254762435260289]
                                 , [-8.14516338216728e-05, -2.18755030455161e-05, 0.12992080460854]
                                 , [4.10335001911372e-07, -5.15301640702395e-08, 0.00170327676596681]])

X_image = 1920
Y_image = 1080
R = 0.031
TIME_INTERVAL = (1 / 240) * 2
WINDOW = 3

P_PLUS_PATH = "Ball videos 3/calibration frames/P_inv.mat"
P_PATH = "Ball videos 3/calibration frames/P.mat"


# todo
# import calibration matrix
# find the pseudo inverse of the camera matrix
# compute the rays of center, right and left
# find the angle between right and left rays
# compute distance to center in equation y = r/size(alpha/2)
# find the coordinates of center


class CenterFinder:
    def __init__(self, radii_path, centers_path):
        start_frame = 0
        self.radii = self.load_radii(radii_path, do_smooth=True)[start_frame:]
        self.centers = np.squeeze(np.load(centers_path))[start_frame:, :]
        # self.display_graph(self.centers)
        # self.display_graph(self.radii)
        self.num_points = self.centers.shape[0]
        self.K = ValMov_K_MATRIX
        self.K_inv = np.linalg.inv(self.K)
        self.P, self.Pinv = self.get_camera_matrix()
        self.right_pnts, self.left_pnts = self.get_right_left_points()
        self.calculate_centers_3D()
        self.calculate_speed()

    def load_radii(self, radii_path, do_smooth=True, window_size=5):
        radii = np.load(radii_path)
        if do_smooth:
            radii = self.do_smoothing(radii, window_size, method='ema', alpha=0.8)
        return radii

    def get_camera_matrix(self):
        P = sio.loadmat(P_PATH)['P_new']
        # Pinv = sio.loadmat(P_PLUS_PATH)['P_inv_new']
        Pinv = MOV3_PINV_MATRIX2
        return P, Pinv

    def get_right_left_points(self):
        left_pnts = np.copy(self.centers)
        right_pnts = np.copy(self.centers)
        left_pnts[:, 1] = left_pnts[:, 1] - self.radii
        right_pnts[:, 1] = right_pnts[:, 1] + self.radii
        return self.switch_xy(right_pnts), self.switch_xy(left_pnts)

    def get_rays(self, points_2D):
        points_2D = self.switch_xy(self.homog(points_2D))
        # points_2D[:, 1] = Y_image - points_2D[:, 1]
        rays = np.array([self.Pinv @ x for x in points_2D])
        # rays /= rays[:, -1].reshape((-1,1))
        return rays

    def calculate_centers_3D(self):
        angles_deg = []
        thetas_rad = []
        distances = []
        centers_3D = []

        # get thetas
        for i in range(self.num_points):
            left_p = self.left_pnts[i]
            right_p = self.right_pnts[i]
            x1, y1 = left_p[1], left_p[0]
            x2, y2 = right_p[1], right_p[0]
            theta_rad = self.opening_angle(x1, x2, y1, y2)
            theta_deg = np.rad2deg(theta_rad)
            angles_deg.append(theta_deg)
            thetas_rad.append(theta_rad)

        thetas_rad = np.array(thetas_rad)
        # thetas_rad = self.do_smoothing(thetas_rad, method='ema', alpha=0.3)
        # thetas_rad = self.do_smoothing(thetas_rad, method='median', window_size=5)
        # get 3d centers
        for i in range(self.num_points):
            theta_rad = thetas_rad[i]
            y = R / np.sin(theta_rad / 2)
            distances.append(y)
            center_2D = self.switch_xy(np.expand_dims(self.centers[i], axis=0))
            center_2D = np.squeeze(self.homog(center_2D))
            center_ray_3D = self.K_inv @ center_2D
            center_ray_3D_n = center_ray_3D / np.linalg.norm(center_ray_3D)
            center_3D = center_ray_3D_n * y
            centers_3D.append(center_3D)

        self.distances = np.array(distances)
        self.angles_deg = np.array(angles_deg)

        self.display_graph(thetas_rad)
        self.centers_3D = np.array(centers_3D)

    def opening_angle(self, u1, v1, u2, v2):
        # Convert the pixel coordinates to homogeneous coordinates (3x1)
        p1 = np.array([[u1], [u2], [1]])
        p2 = np.array([[v1], [v2], [1]])

        # Invert the intrinsic matrix K
        # self.K_inv = np.linalg.inv(self.K)

        # Convert the homogeneous coordinates to camera coordinates (3x1)
        c1 = self.K_inv @ p1
        c2 = self.K_inv @ p2

        # Normalize the camera coordinates to get unit vectors (3x1)
        u1 = c1 / np.linalg.norm(c1)
        u2 = c2 / np.linalg.norm(c2)

        # Calculate the dot product of the unit vectors
        dot_product = np.dot(u1.T, u2)

        # Calculate the angle between the unit vectors in radians
        angle = math.acos(dot_product)

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle)

        return angle

    def calculate_speed(self):
        locations = self.centers_3D
        # locations = self.do_smoothing(locations,window_size=3, method='median')
        distances = np.linalg.norm(locations[WINDOW:] - locations[:-WINDOW], axis=1)
        # Calculate the speed between each pair of consecutive points
        self.speeds = distances / (TIME_INTERVAL * WINDOW)
        abs_speeds = np.abs(self.speeds)
        # Calculate the change in speed between each pair of consecutive points
        changes_in_speed = self.speeds[1:] - self.speeds[:-1]
        # Calculate the acceleration between each pair of consecutive points
        accelerations = changes_in_speed / TIME_INTERVAL
        abs_accelerations = np.abs(accelerations)

        # self.display_graph(locations)
        self.display_graph(self.speeds, title="Speed vs Frames", X_label='frames', Y_label='speed (m/s)')
        # self.display_graph(abs_accelerations)
        pass

    def display_3D_trajectory(self):
        # Separate the coordinates
        # Fit the data into a 3D line using PCA
        pca = PCA(n_components=2)
        pca.fit(self.centers_3D)

        # The coefficients of the first principal component vector
        coeff_x, coeff_y, coeff_z = pca.components_[0]

        # The mean of the data
        mean_x, mean_y, mean_z = pca.mean_

        # Create a range of values for t
        t_range = np.linspace(-0.5, 0.5, 100)

        # Create the 3D line
        x_line = mean_x + coeff_x * t_range
        y_line = mean_y + coeff_y * t_range
        z_line = mean_z + coeff_z * t_range

        # Create a new figure for plotting
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Plot the original 3D trajectory
        ax.plot(self.centers_3D[:, 0], self.centers_3D[:, 1], self.centers_3D[:, 2], marker='o', color='red', markersize=5)

        # Plot the fitted line
        # ax.plot(x_line, y_line, z_line, color='blue')

        # Set some properties of the axes object
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D Trajectory and Fitted Line')

        # Show the graph
        plt.show()

        # Calculate distances from points to the line and average them
        distances = np.abs((self.centers_3D[:, 0] - mean_x) * coeff_x + (self.centers_3D[:, 1] - mean_y) * coeff_y + (
                    self.centers_3D[:, 2] - mean_z) * coeff_z) / np.sqrt(coeff_x ** 2 + coeff_y ** 2 + coeff_z ** 2)
        average_distance = np.mean(distances)
        print('Average distance:', average_distance)
    @staticmethod
    def do_smoothing(data, window_size=5, method='median', alpha=0.5):
        if len(data.shape) == 1: data = np.expand_dims(data, axis=1)
        num_dims = data.shape[1]
        data_smoothed = np.zeros(data.shape)
        for dim in range(num_dims):
            if method == 'median':
                data_smoothed[:, dim] = pd.Series(data[:, dim]).rolling(window_size).median()
                data_smoothed[:window_size, dim] = data[:window_size, dim]
            elif method == 'ema':
                data_smoothed[:, dim] = pd.Series(data[:, dim]).ewm(alpha=alpha).mean()
        return np.squeeze(data_smoothed)

    @staticmethod
    def display_graph(array, title='', X_label='', Y_label=''):
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.title(title)
        plt.plot(array)
        plt.show()

    @staticmethod
    def normalize_homog(array):
        return array / array[:, -1].reshape((-1, 1))

    @staticmethod
    def switch_xy(array):  # array of size (N,2/3)
        A = np.copy(array[:, 0])
        B = np.copy(array[:, 1])
        array[:, 0] = B
        array[:, 1] = A
        return array

    @staticmethod
    def homog(array):  # array of size (N,2)
        ones = np.ones((array.shape[0], 1))  # Create a column of zeros with M rows and 1 column
        homog = np.c_[array, ones]
        return homog


if __name__ == '__main__':
    CF = CenterFinder(radii_path=r'ball videos 4/radii_throw_2_skip_2.npy',
                      centers_path='ball videos 4/centers_throw_2_skip_2.npy')

    CF.display_3D_trajectory()
