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


def triangulate(k1, k2, r1, r2, t1, t2, matches):
    """
    Arguments:
        k1: intrinsic matrix for image 1, dim: (3, 3)
        k2: intrinsic matrix for image 2, dim: (3, 3)
        r1: rotation matrix for image 1, dim: (3, 3)
        r2: rotation matrix for image 1, dim: (3, 3)
        t1: translation for image 1, dim: (3,)
        t2: translation for image 1, dim: (3,)
        matches:  dim: (#matches, 4)
    Returns:
        points_3d, dim: (#matches, 3)
    """
    # --------------------------- Begin your code here ---------------------------------------------
    pts_3d = []
    n, *_ = matches.shape
    p1 = k1 @ np.hstack([r1, t1.reshape(-1, 1)])
    p2 = k2 @ np.hstack([r2, t2.reshape(-1, 1)])

    for i in range(n):
        x, y, x_prime, y_prime = matches[i]
        a = np.vstack([
            y * p1[2] - p1[1],
            p1[0] - x * p1[2],
            y_prime * p2[2] - p2[1],
            p2[0] - x_prime * p2[2]
        ])
        *_, vh = np.linalg.svd(a, full_matrices=True)
        pts_3d += [vh[-1]]
    pts_3d = np.stack(pts_3d, axis=0)
    pts_3d = pts_3d[:, :3] / pts_3d[:, 3].reshape(-1, 1)
    # --------------------------- End your code here   ---------------------------------------------
    return pts_3d


if __name__ == "__main__":
    # Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
    # For each row, it consists (k1_x, k1_y, k2_x, k2_y).
    # If necessary, you can convert float to int to get the integer coordinate
    all_good_matches = np.load('../assets/all_good_matches.npy')

    K1 = np.load('../assets/fountain/Ks/0000.npy')
    K2 = np.load('../assets/fountain/Ks/0005.npy')

    R1 = np.load('../assets/fountain/Rs/0000.npy')
    R2 = np.load('../assets/fountain/Rs/0005.npy')

    T1 = np.load('../assets/fountain/ts/0000.npy')
    T2 = np.load('../assets/fountain/ts/0005.npy')

    points_3d = triangulate(K1, K2, R1, R2, T1, T2, all_good_matches)
    if points_3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        # Visualize both point and camera
        # Check this link for Open3D visualizer
        # http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
        # Check this function for adding a virtual camera in the visualizer
        # Open3D is not the only option. You could use matplotlib, vtk or other visualization tools as well.
        # --------------------------- Begin your code here ---------------------------------------------

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R1
        extrinsic[:3, -1] = T1.reshape(-1)
        cam1 = o3d.geometry.LineSet().create_camera_visualization(1520, 1006, K1, extrinsic)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R2
        extrinsic[:3, -1] = T2.reshape(-1)
        cam2 = o3d.geometry.LineSet().create_camera_visualization(1520, 1006, K2, extrinsic)

        o3d.visualization.draw_geometries([pcd, cam1, cam2])
        # --------------------------- End your code here   ---------------------------------------------
