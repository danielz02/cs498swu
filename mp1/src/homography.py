import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# You could pip install the following dependencies if any is missing
# pip install -r requirements.txt

# Load the image and plot the keypoints
im = imread('uiuc.png') / 255.0
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

corners_court = []

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
    - directly calling findHomography in cv2 will receive zero point, but you could use it as sanity-check of your own implementation
'''


def find_homography(pts_src, pts_dst):
    # --------------------------- Begin your code here ---------------------------------------------
    pass


# --------------------------- End your code here   ---------------------------------------------

# Calculate the homography matrix using your implementation
H = find_homography(corners_court, keypoints_im)

'''
Question 3.a: insert the logo virtually onto the state farm center image.
Specific requirements:
     - the size of the logo needs to be 3x6 meters;
     - the bottom left logo corner is at the location (23, 2) on the basketball court.
Returns:
     transform_target - The transformation matrix from logo.png image coordinate to target.png coordinate (3x3 matrix)

Hints:
     - Consider to calculate the transform as the composition of the two: H_logo_target = H_court_target @ H_logo_court
     - Given the banner size in meters and image size in pixels, could you scale the logo image coordinate from pixels to meters
     - What transform will move the logo to the target location?
     - Could you leverage the homography between basketball court to target image we computed in Q.2?
     - Image coordinate is y down ((0, 0) at bottom-left corner) while we expect the inserted logo to be y up, how would you handle this?
'''

# Read the banner image that we want to insert to the basketball court
logo = imread('logo.png') / 255.0
plt.clf()
plt.imshow(logo)
plt.title("Banner")
plt.show()

# --------------------------- Begin your code here ---------------------------------------------

target_transform = []

# --------------------------- End your code here   ---------------------------------------------

'''
Question 3.b: compute the warpImage function
Arguments:
     image - the source image you may want to warp (Hs x Ws x 4 matrix, R,G,B,alpha)
     H - the homography transform from the source to the target image coordinate (3x3 matrix)
     shape - a tuple of the target image shape (Wt, Ht)
Returns:
     image_warped - the warped image (Ht x Wt x 4 matrix)

Hints:
    - you might find the function numpy.meshgrid and numpy.ravel_multi_index useful;
    - are you able to get rid of any for-loop over all pixels?
    - directly calling warpAffine or warpPerspective in cv2 will receive zero point, but you could use as sanity-check of your own implementation
'''


def warp_image(image, H, shape):
    # --------------------------- Begin your code here ---------------------------------------------
    pass


# --------------------------- End your code here   ---------------------------------------------

# call the warpImage function
logo_warp = warp_image(logo, target_transform, (im.shape[1], im.shape[0]))

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

im = []

# --------------------------- End your code here   ---------------------------------------------

plt.clf()
plt.imshow(im)
plt.title("Blended Image")
plt.show()

# dump the results for autograde
outfile = 'solution_homography.npz'
np.savez(outfile, corners_court, H, target_transform, logo_warp, im)
