import numpy as np
import cv2 
import math

class RealDistance:
    def __init__(self, pixel_array, real_point_array):
        self.homography_pixel = pixel_array
        self.real_world_pixel = real_point_array


    def calculate_homography(self):

        H, _ = cv2.findHomography(self.homography_pixel, self.real_world_pixel)
        return H

    def calculate_real_world_distance(self, H_inv, p1_pixel, p2_pixel):

        # Convert pixel points to homogeneous coordinates
        p1_homo = np.append(p1_pixel, 1)
        p2_homo = np.append(p2_pixel, 1)

        # Transform the points using the inverse homography matrix
        p1_real_homo = np.dot(H_inv, p1_homo)
        p2_real_homo = np.dot(H_inv, p2_homo)

        # Normalize the real-world coordinates
        p1_real_x = p1_real_homo[0] / p1_real_homo[2]
        p1_real_y = p1_real_homo[1] / p1_real_homo[2]
        p2_real_x = p2_real_homo[0] / p2_real_homo[2]
        p2_real_y = p2_real_homo[1] / p2_real_homo[2]

        delta_x = p2_real_x - p1_real_x
        delta_y = p2_real_y - p1_real_y
        distance = math.sqrt(delta_x**2 + delta_y**2)

        return distance