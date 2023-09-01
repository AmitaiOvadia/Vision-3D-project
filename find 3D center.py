import scipy.io as sio
import numpy as np
import pygame
import math


# one estimation formula can be: distance = size_obj * focal_length / size_obj_on_sensor

MOV3_K_MATRIX = np.array([[1.5873e03,  0, 960.9],
                          [0, 1.5757e3, 561.8347],
                          [0, 0,        1]])


MOV3_P_MATRIX = np.array([[1.5873e03,  0, 960.9,  0],
                          [0,          1.5757e3,  561.8347, 0],
                          [0,          0,         1, 0]])

MOV3_PINV_MATRIX1 = np.linalg.pinv(MOV3_P_MATRIX)

MOV3_PINV_MATRIX2 = np.array([[-0.000593920071308024,	-4.91334853074069e-05,	0.892119023580987]
                     ,[4.18938354926008e-05,	-0.000631928273524856,	0.254762435260289]
                     ,[-8.14516338216728e-05,	-2.18755030455161e-05,	0.12992080460854]
                     ,[4.10335001911372e-07,	-5.15301640702395e-08,	0.00170327676596681]])
X_image = 1920
Y_image = 1080
R = 0.03
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
        self.radii = np.load(radii_path)
        self.centers = np.load(centers_path)
        self.num_points = self.centers.shape[0]
        self.K = MOV3_K_MATRIX
        self.P, self.Pinv = self.get_camera_matrix()
        self.right_pnts, self.left_pnts = self.get_right_left_points()
        self.right_rays, self.left_rays = self.get_rays(self.right_pnts), self.get_rays(self.left_pnts)
        self.get_angles()
        self.get_centers_distances()

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
        return right_pnts, left_pnts

    def get_rays(self, points_2D):
        points_2D = self.switch_xy(self.homog(points_2D))
        # points_2D[:, 1] = Y_image - points_2D[:, 1]
        rays = np.array([self.Pinv @ x for x in points_2D])
        # rays /= rays[:, -1].reshape((-1,1))
        return rays

    def get_angles(self):
        for i in range(self.num_points):
            left_p = self.left_pnts[i]
            right_p = self.right_pnts[i]
            u1, u2 = left_p[1], left_p[0]
            v1, v2 = right_p[1], right_p[0]
            theta_rad = self.opening_angle(u1, v1, u2, v2)
            theta_deg = np.rad2deg(theta_rad)
            y = R / np.sin(theta_rad/2)


    def opening_angle(self, u1, v1, u2, v2):
        # Convert the pixel coordinates to homogeneous coordinates (3x1)
        p1 = np.array([[u1], [u2], [1]])
        p2 = np.array([[v1], [v2], [1]])

        # Invert the intrinsic matrix K
        K_inv = np.linalg.inv(self.K)

        # Convert the homogeneous coordinates to camera coordinates (3x1)
        c1 = K_inv @ p1
        c2 = K_inv @ p2

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








    @staticmethod
    def normalize_homog(array):
        return array / array[:, -1].reshape((-1,1))

    @staticmethod
    def switch_xy(array): # array of size (N,2/3)
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
    CF = CenterFinder(radii_path=r'Ball videos 3\radii_throw_2.npy',
                      centers_path='Ball videos 3\centers_throw_2.npy')
    a=0
