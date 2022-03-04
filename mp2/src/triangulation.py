"""
Question 5. Triangulation
In this question we move to 3D.
You are given keypoint matching between two images, together with the camera intrinsic and extrinsic matrix.
Your task is to perform triangulation to restore the 3D coordinates of the key points.
In your PDF, please visualize the 3d points and camera poses in 3D from three different viewing perspectives.
"""
import os
import cv2  # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
all_good_matches = np.load('../assets/all_good_matches.npy')

K1 = np.load('../assets/fountain/Ks/0000.npy')
K2 = np.load('../assets/fountain/Ks/0005.npy')

R1 = np.load('../assets/fountain/Rs/0000.npy')
R2 = np.load('../assets/fountain/Rs/0005.npy')

t1 = np.load('../assets/fountain/ts/0000.npy')
t2 = np.load('../assets/fountain/ts/0005.npy')


def triangulate(K1, K2, R1, R2, t1, t2, all_good_matches):
    """
    Arguments:
        K1: intrinsic matrix for image 1, dim: (3, 3)
        K2: intrinsic matrix for image 2, dim: (3, 3)
        R1: rotation matrix for image 1, dim: (3, 3)
        R2: rotation matrix for image 1, dim: (3, 3)
        t1: translation for image 1, dim: (3,)
        t2: translation for image 1, dim: (3,)
        all_good_matches:  dim: (#matches, 4)
    Returns:
        points_3d, dim: (#matches, 3)
    """
    points_3d = None
    # --------------------------- Begin your code here ---------------------------------------------

    # --------------------------- End your code here   ---------------------------------------------
    return points_3d


points_3d = triangulate(K1, K2, R1, R2, t1, t2, all_good_matches)
if points_3d is not None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Visualize both point and camera Check this link for Open3D visualizer
    # http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries Check
    # this function for adding a virtual camera in the visualizer
    # http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries Open3D is
    # not the only option. You could use matplotlib, vtk or other visualization tools as well.
    # --------------------------- Begin your code here ---------------------------------------------

    # --------------------------- End your code here   ---------------------------------------------
