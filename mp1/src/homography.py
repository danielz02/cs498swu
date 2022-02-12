import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# You could pip install the following dependencies if any is missing
# pip install -r requirements.txt

# Load the image and plot the keypoints
im = imread('../img/uiuc.png') / 255.0
keypoints_im = np.array([(604.593078169188, 583.1361439828671),
                         (1715.3135416380655, 776.304920238324),
                         (1087.5150188078305, 1051.9034760165837),
                         (79.20731171576836, 642.2524505093215)])

print(keypoints_im)
plt.clf()
plt.imshow(im)
plt.scatter(keypoints_im[:, 0], keypoints_im[:, 1])
plt.plot(keypoints_im[[0, 1, 2, 3, 0], 0], keypoints_im[[0, 1, 2, 3, 0], 1], 'g')

for ind, corner in enumerate(keypoints_im):
    plt.text(corner[0] + 30.0, corner[1] + 30.0, '#' + str(ind),
             c='b', family='sans-serif', size='x-large')
plt.title("Target Image and Keypoints")
plt.show()

'''
Question 1: specify the corners' coordinates
Take point 3 as origin, the long edge as x axis and short edge as y axis
Output:
     - corners_court: a numpy array (4x2 matrix)
'''
# --------------------------- Begin your code here ---------------------------------------------

corners_court = np.array([
    [0, 15.24],
    [28.65, 15.24],
    [28.65, 0],
    [0, 0]
])

# --------------------------- End your code here   ---------------------------------------------

'''
Question 2: complete the findHomography function
Arguments:
     pts_src - Each row corresponds to an actual point on the 2D plane (Nx2 matrix)
     pts_dst - Each row is the pixel location in the target image coordinate (Nx2 matrix)
Returns:
     H - The homography matrix (3x3 matrix)

Hints:
    - you might find the function vstack, hstack to be handy for getting homogenous coordinate;
    - you might find numpy.linalg.svd to be useful for solving linear system
    - directly calling findHomography in cv2 will receive zero point, but you could use it as sanity-check of your
    own implementation
'''


def findHomography(pts_src, pts_dst):
    # --------------------------- Begin your code here ---------------------------------------------
    n, *_ = pts_src.shape
    pts_src = np.hstack([pts_src, np.ones((n, 1))])
    pts_dst = np.hstack([pts_dst, np.ones((n, 1))])

    a = np.zeros((2 * n, 9))
    a[0::2, 3:6] = pts_src
    a[1::2, 0:3] = pts_src
    a[0::2, 6:9] = -pts_src * pts_dst[:, 1].reshape(-1, 1)
    a[1::2, 6:9] = -pts_src * pts_dst[:, 0].reshape(-1, 1)
    *_, vh = np.linalg.svd(a)
    h = vh[-1]
    h /= h[-1]

    return h.reshape(3, 3)


# --------------------------- End your code here   ---------------------------------------------

# Calculate the homography matrix using your implementation
H_court_target = findHomography(corners_court, keypoints_im)

'''
Question 3.a: insert the logo virtually onto the state farm center image.
Specific requirements:
     - the size of the logo needs to be 3x6 meters;
     - the bottom left logo corner is at the location (23, 2) on the basketball court.
Returns:
     transform_target - The transformation matrix from logo.png image coordinate to target.png coordinate (3x3 matrix)

Hints:
     - Consider to calculate the transform as the composition of the two: H_logo_target = H_court_target @ H_logo_court
     - Given the banner size in meters and image size in pixels, could you scale the logo image coordinate from pixels
     to meters
     - What transform will move the logo to the target location?
     - Could you leverage the homography between basketball court to target image we computed in Q.2?
     - Image coordinate is y down ((0, 0) at bottom-left corner) while we expect the inserted logo to be y up, how would
     you handle this?
'''

# Read the banner image that we want to insert to the basketball court
logo = imread('../img/logo.png') / 255.0
plt.clf()
plt.imshow(logo)
plt.title("Banner")
plt.show()

# --------------------------- Begin your code here ---------------------------------------------

corners_logo = np.array([
    [23, 2 + 3],
    [23 + 6, 2 + 3],
    [23 + 6, 2],
    [23, 2]
])

h_logo, w_logo, _ = logo.shape
pts_logo = np.array([
    [0, 0],
    [0, w_logo],
    [h_logo, w_logo],
    [h_logo, 0]
])

H_logo_court = findHomography(pts_logo, corners_logo)

coordinate_conversion = np.array([
    [-1, 0, 4 * h_logo],
    [0, 1, 0],
    [0, 0, 1]
])

target_transform = coordinate_conversion @ H_court_target @ H_logo_court

# --------------------------- End your code here   ---------------------------------------------

'''
Question 3.b: compute the warp_image function
Arguments:
     image - the source image you may want to warp (Hs x Ws x 4 matrix, R, G, B, alpha)
     H - the homography transform from the source to the target image coordinate (3x3 matrix)
     shape - a tuple of the target image shape (Wt, Ht)
Returns:
     image_warped - the warped image (Ht x Wt x 4 matrix)

Hints:
    - you might find the function numpy.meshgrid and numpy.ravel_multi_index useful;
    - are you able to get rid of any for-loop over all pixels?
    - directly calling warpAffine or warpPerspective in cv2 will receive zero point, but you could use as sanity-check
    of your own implementation
'''


def warpImage(image, h, shape):
    # --------------------------- Begin your code here ---------------------------------------------
    h_src, w_src, _ = image.shape
    xs, ys = np.meshgrid(np.arange(0, w_src), np.arange(0, h_src), indexing="xy")
    coords_src = np.stack([xs, ys, np.ones_like(xs)], axis=0)
    coords_target = h @ coords_src.reshape(3, -1)
    coords_target /= coords_target[-1]
    coords_target = coords_target[:-1].astype(int)
    print(coords_target)

    # coords_target = np.ravel_multi_index(coords_target, dims=shape[:2])

    image_warp = np.zeros((*shape, 4))
    image_warp[coords_target[0], coords_target[1], :] = image.reshape(-1, 4)

    return image_warp.transpose((1, 0, 2))


# --------------------------- End your code here   ---------------------------------------------

# call the warpImage function
logo_warp = warpImage(logo, target_transform, (im.shape[1], im.shape[0]))

plt.clf()
plt.imshow(logo_warp)
plt.title("Warped Banner")
plt.show()

'''
Question 3.c: alpha-blend the warped logo and state farm center image

im = logo * alpha_logo + target * (1 - alpha_logo)

Hints:
    - try to avoid for-loop. You could either use numpy's tensor broadcasting or explicitly call np.repeat / np.tile
'''

# --------------------------- Begin your code here ---------------------------------------------

alpha_logo = 0.35
im = alpha_logo * logo_warp + im * (1 - alpha_logo)

# --------------------------- End your code here   ---------------------------------------------

plt.clf()
plt.imshow(im)
plt.title("Blended Image")
plt.show()

# dump the results for autograde
outfile = '../img/solution_homography.npz'
np.savez(outfile, corners_court, H_court_target, target_transform, logo_warp, im)
