"""
Questions 2-4. Fundamental matrix estimation

Question 2. Eight-point Estimation For this question, your task is to implement normalized and unnormalized
eight-point algorithms to find out the fundamental matrix between two cameras. We've provided a method to compute the
average geometric distance, which is the distance between each projected keypoint from one image to its corresponding
epipolar line in the other image. You might consider reading that code below as a reminder for how we can use the
fundamental matrix. For more information on the normalized eight-point algorithm, please see this link:
https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm

Question 3. RANSAC Your task is to implement RANSAC to find out the fundamental matrix between two cameras if the
correspondences are noisy.

Please report the average geometric distance based on your estimated fundamental matrix, given 1, 100,
and 10000 iterations of RANSAC. Please also visualize the inliers with your best estimated fundamental matrix in your
solution for both images (we provide a visualization function). In your PDF, please also explain why we do not
perform SVD or do a least-square over all the matched key points.

Question 4. Visualizing Epi-polar Lines
Please visualize the epi-polar line for both images for your estimated F in Q2 and Q3.

To draw it on images, cv2.line, cv2.circle are useful to plot lines and circles.
Check our Lecture 4, Epipolar Geometry, to learn more about equation of epipolar line.
Our Lecture 4 and 5 cover most of the concepts here.
This link also gives a thorough review of epipolar geometry:
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
"""

import os
import cv2  # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random
from pathlib import Path


# --------------------- Question 2

def calculate_geometric_distance(all_matches, F):
    """
    Calculate average geometric distance from each projected keypoint from one image to its corresponding epi-polar
    line in another image. Note that you should take the average of the geometric distance in two direction (image 1
    to 2, and image 2 to 1) Arguments: all_matches: all matched keypoint pairs that loaded from disk (#all_matches,
    4). F: estimated fundamental matrix, (3, 3) Returns: average geometric distance.
    """
    ones = np.ones((all_matches.shape[0], 1))
    all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
    # Epi-polar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    # Final distance.
    dist1 = d1.sum() / all_matches.shape[0]
    dist2 = d2.sum() / all_matches.shape[0]

    dist = (dist1 + dist2) / 2
    return dist


def estimate_fundamental_matrix(matches, normalize=False):
    """
    Arguments:
        matches: Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4).
        normalize: Boolean flag for using normalized or unnormalized alg.
    Returns:
        F: Fundamental matrix, dims (3, 3).
    """
    # --------------------------- Begin your code here ---------------------------------------------
    n = len(matches)
    img1_matches = matches[:, :2]
    img2_matches = matches[:, 2:]
    trans_a, trans_b = np.eye(3), np.eye(3)
    if normalize:
        img1_matches_centroid = img1_matches.mean(axis=0)
        img2_matches_centroid = img2_matches.mean(axis=0)
        img1_matches = img1_matches - img1_matches_centroid
        img2_matches = img2_matches - img2_matches_centroid

        img1_scale = np.sqrt(2 / (1 / n * np.sum(img1_matches ** 2)))
        img2_scale = np.sqrt(2 / (1 / n * np.sum(img2_matches ** 2)))

        img1_matches = img1_matches * img1_scale
        img2_matches = img2_matches * img2_scale

        trans_a = np.array([
            [img1_scale, 0, -img1_scale * img1_matches_centroid[0]],
            [0, img1_scale, -img1_scale * img1_matches_centroid[1]],
            [0, 0, 1]
        ])
        trans_b = np.array([
            [img2_scale, 0, -img2_scale * img2_matches_centroid[0]],
            [0, img2_scale, -img2_scale * img2_matches_centroid[1]],
            [0, 0, 1]
        ])

    img1_matches = np.hstack([img1_matches, np.ones(n).reshape(-1, 1)])
    img2_matches = np.hstack([img2_matches, np.ones(n).reshape(-1, 1)])

    x, x_prime = img1_matches[:, 0].reshape(-1, 1), img2_matches[:, 0].reshape(-1, 1)
    y, y_prime = img1_matches[:, 1].reshape(-1, 1), img2_matches[:, 1].reshape(-1, 1)
    u = np.hstack([x_prime * x, x_prime * y, x_prime, y_prime * x, y_prime * y, y_prime, x, y,
                   (img1_matches[:, 2] * img2_matches[:, 2]).reshape(-1, 1)])
    u, s, vh = np.linalg.svd(u)
    f = vh[-1].reshape(3, 3)
    u_f, s_f, vh_f = np.linalg.svd(f, compute_uv=True, full_matrices=True)
    s_f[-1] = 0
    f = u_f @ np.diag(s_f) @ vh_f

    if normalize:
        f = trans_b.T @ f @ trans_a

    # --------------------------- End your code here   ---------------------------------------------
    return f


# --------------------- Question 3

def ransac(all_matches, max_iterations, estimate_fundamental, threshold):
    """
    Arguments:
        all_matches: coords of matched keypoint pairs in image 1 and 2, dims (# matches, 4).
        max_iterations: total number of RANSAC iteration
        estimate_fundamental: your eight-point algorithm function but use normalized version
        threshold: threshold to decide if one point is inlier
    Returns:
        best_f: the best Fundamental matrix, dims (3, 3).
        inlier_matches_with_best_f: (#inliers, 4)
        avg_geo_dis_with_best_f: float
    """

    best_f = np.eye(3)
    inlier_matches_with_best_f = []
    avg_geo_dis_with_best_f = np.inf

    # --------------------------- Begin your code here ---------------------------------------------

    for _ in range(max_iterations):
        # random sample correspondences
        random_matches = all_matches[np.random.choice(range(len(all_matches)), size=8, replace=False)]

        # estimate the minimal fundamental estimation problem
        f = estimate_fundamental(random_matches, normalize=True)

        # compute # of inliers
        dists = np.array([calculate_geometric_distance(match.reshape(1, -1), f) for match in random_matches])
        inliers = random_matches[dists < threshold]
        avg_geo_dis = dists[dists < threshold].mean()

        # update the current best solution

        if avg_geo_dis < avg_geo_dis_with_best_f:
            avg_geo_dis_with_best_f = avg_geo_dis
            inlier_matches_with_best_f = inliers.copy()
            best_f = f.copy()

    # --------------------------- End your code here   ---------------------------------------------
    return best_f, inlier_matches_with_best_f, avg_geo_dis_with_best_f


def visualize_inliers(im1, im2, inlier_coords):
    for i, im in enumerate([im1, im2]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(im, cmap='gray')
        plt.scatter(inlier_coords[:, 2 * i], inlier_coords[:, 2 * i + 1], marker="x", color="red", s=10)
    plt.show()


# --------------------- Question 4

def visualize(f, im1, im2, m1, m2):
    # --------------------------- Begin your code here ---------------------------------------------
    n, *_ = m1.shape
    h, w, *_ = im1.shape
    m1 = np.hstack([m1, np.ones(n).reshape(-1, 1)])
    m2 = np.hstack([m2, np.ones(n).reshape(-1, 1)])
    l_prime = np.array([f.T @ x for x in m1])
    l = np.array([f @ x_prime for x_prime in m2])

    l_slope = - l[:, 0] / l[:, 1]
    l_intercept = - l[:, 2] / l[:, 1]
    l_prime_slope = - l_prime[:, 0] / l_prime[:, 1]
    l_prime_intercept = - l_prime[:, 2] / l_prime[:, 1]

    _, ax = plt.subplots(ncols=2)
    for i in range(n):
        ax[0].plot(
            np.array([0, w]), np.array([0, w]) * l_slope[i] + l_intercept[i], color="C0", linewidth=1
        )
        ax[1].plot(
            np.array([0, w]), np.array([0, w]) * l_prime_slope[i] + l_prime_intercept[i], color="C0", linewidth=1
        )

    for i in range(n):
        ax[0].add_patch(plt.Circle(m1[i][:2], radius=10, color="red"))
        ax[1].add_patch(plt.Circle(m2[i][:2], radius=10, color="red"))

    # ax[0].set_ylim([h, 0])
    # ax[1].set_ylim([h, 0])
    ax[0].imshow(im1, cmap="gray")
    ax[1].imshow(im2, cmap="gray")
    plt.show()

    # --------------------------- End your code here   ---------------------------------------------


if __name__ == "__main__":
    basedir = Path('../assets/fountain')
    img1 = cv2.imread(str(basedir / 'images/0000.png'), 0)
    img2 = cv2.imread(str(basedir / 'images/0005.png'), 0)

    fig, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img1, cmap='gray')
    axarr[1].imshow(img2, cmap='gray')
    plt.show()

    # Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
    # For each row, it consists (k1_x, k1_y, k2_x, k2_y).
    # If necessary, you can convert float to int to get the integer coordinate
    eight_good_matches = np.load('../assets/eight_good_matches.npy')
    all_good_matches = np.load('../assets/all_good_matches.npy')

    F_with_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=True)
    F_without_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=False)

    # Evaluation (these numbers should be quite small)
    print(
        f"F_with_normalization average geo distance: "
        f"{calculate_geometric_distance(all_good_matches, F_with_normalization)}"
    )
    print(
        f"F_without_normalization average geo distance: "
        f"{calculate_geometric_distance(all_good_matches, F_without_normalization)}"
    )

    num_iterations = [1, 100, 10000]
    best_F = np.eye(3)
    inlier_threshold = 0.6  # TODO: change the inlier threshold by yourself
    for num_iteration in num_iterations:
        best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F = ransac(
            all_good_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold
        )
        if inlier_matches_with_best_F is not None:
            print(f"num_iterations: {num_iteration}; avg_geo_dis_with_best_F: {avg_geo_dis_with_best_F};")
            visualize_inliers(img1, img2, inlier_matches_with_best_F)

    all_good_matches = np.load('../assets/all_good_matches.npy')
    F_Q2 = F_without_normalization  # link to your estimated F in Q3
    F_Q3 = best_F  # link to your estimated F in Q3
    visualize(F_Q2, img2, img1, all_good_matches[:, 2:], all_good_matches[:, :2])
    visualize(F_Q3, img2, img1, all_good_matches[:, 2:], all_good_matches[:, :2])
