"""
Question 1. Keypoint Matching - Putative Matches

Your task is to select putative matches between detected SIFT keypoints found in two images.
The matches should be selected based on the Euclidean distance for the pairwise descriptor.
In your implementation, make sure to filter your keypoint matches using Lowe's ratio test
    A great explanation here: https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work

For this question, you should implement your solution without using third-party functions e.g., cv2 knnmatch,
which can significantly trivialize the problem.

Meanwhile, please avoid solutions using for-loop which slows down the calculations.
Hint: To avoid using for-loop, check out scipy.spatial.distance.cdist(X,Y,'sqeuclidean')
"""
import os
import cv2  # our tested version is 4.5.5
import numpy as np
import scipy.spatial.distance
from matplotlib import pyplot as plt
from pathlib import Path


def plot_matches(ax_, img1_, img2_, kp_matches_):
    """
    plot the match between two image according to the matched keypoints
    :param ax_: plot handle
    :param img1_: left image
    :param img2_: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1_, img2_])
    ax_.set_aspect('equal')
    ax_.imshow(res, cmap='gray')

    ax_.plot(kp_matches_[:, 0], kp_matches_[:, 1], '+r')
    ax_.plot(kp_matches_[:, 2] + img1_.shape[1], kp_matches_[:, 3], '+r')
    ax_.plot([kp_matches_[:, 0], kp_matches_[:, 2] + img1_.shape[1]],
             [kp_matches_[:, 1], kp_matches_[:, 3]], 'r', linewidth=0.8)
    ax_.axis('off')


def select_putative_matches(des1_, des2_, ratio_threshold=0.5):
    """
    Arguments:
        :param des1_: cv2 SIFT descriptors extracted from image1 (None x 128 matrix)
        :param des2_: cv2 SIFT descriptors extracted from image2 (None x 128 matrix)
        :param ratio_threshold: threshold for Lowe's ratio test
    Returns:
        matches: List of tuples. Each tuple characterizes a match between descriptor in des1 and descriptor in des2. 
                 The first item in each tuple stores the descriptor index in des1, and the second stores index in des2.
                 For example, returning a tuple (8,10) means des1[8] and des2[10] are a match.
    """

    dists = scipy.spatial.distance.cdist(des1_, des2_, "sqeuclidean")
    dist_ranks = dists.argsort(axis=0)
    fc = np.argwhere(dist_ranks == 0)
    fs = np.argwhere(dist_ranks == 1)
    dist_ratios = dists[fc[:, 0], fc[:, 1]] / dists[fs[:, 0], fs[:, 1]]
    filtered_matches = [tuple(x) for x in fc[dist_ratios < ratio_threshold]]

    return filtered_matches


if __name__ == "__main__":
    basedir = Path('../assets/fountain')
    img1 = cv2.imread(str(basedir / 'images/0000.png'), 0)
    img2 = cv2.imread(str(basedir / 'images/0005.png'), 0)

    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img1, cmap='gray')
    axarr[1].imshow(img2, cmap='gray')
    plt.show()

    # Initiate SIFT detector, The syntax of importing sift descriptor depends on your cv2 version. The following is for
    # cv2 4.5.5,
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # Hints: kp1 and des1 has the same order. kp1.pt gives the x-y coordinate
    kp1, des1 = sift.detectAndCompute(img1, None)
    # Hints: kp2 and des2 has the same order. kp2.pt gives the x-y coordinate
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = select_putative_matches(des1, des2)  # your code
    match_indices = np.random.permutation(len(matches))[:20]  # you can change it to visualize more points
    kp_matches = np.array(
        [[kp1[i].pt[0], kp1[i].pt[1], kp2[j].pt[0], kp2[j].pt[1]] for i, j in np.array(matches)[match_indices]]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_matches(ax, img1, img2, kp_matches)
    ax.set_title('Matches visualization')
    plt.show()
