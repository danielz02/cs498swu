import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from numba import njit


@njit()
def stereo_matching_ssd(left_im, right_im, max_disp=128, block_size=7):
    """
    Using sum of square difference to compute stereo matching.
    Arguments:
        left_im: left image (h x w x 3 numpy array)
        right_im: right image (h x w x 3 numpy array)
        max_disp: maximum possible disparity
        block_size: size of the block for computing matching cost
    Returns:
        disp_im: disparity image (h x w numpy array), storing the disparity values
  """
    # --------------------------- Begin your code here ---------------------------------------------
    assert left_im.shape == right_im.shape
    im_h, im_w, _ = left_im.shape

    block_half = block_size // 2
    disp_map = np.zeros_like(left_im[:, :, 0])
    # We need to slide horizontally
    for y in range(im_h):
        for x in range(im_w):
            best_ssd = np.inf
            best_disp = np.inf
            for offset in range(max_disp):
                y_lower, y_upper = max(y - block_half, 0),  min(y + block_half, im_h)
                left_patch = left_im[y_lower:y_upper, max(x - block_half, 0):min(x + block_half, im_w), :]
                right_patch = right_im[y_lower:y_upper, max(x - block_half - offset, 0):min(x + block_half - offset, im_w), :]
                if left_patch.shape != right_patch.shape:
                    continue
                ssd = np.sum((left_patch - right_patch) ** 2)
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_disp = offset
            disp_map[y, x] = best_disp * (255 / max_disp)

    # --------------------------- End your code here   ---------------------------------------------
    return disp_map


if __name__ == "__main__":
    # read intrinsics, extrinsics and camera images
    K1 = np.load('../assets/fountain/Ks/0005.npy')
    K2 = np.load('../assets/fountain/Ks/0004.npy')
    R1 = np.load('../assets/fountain/Rs/0005.npy')
    R2 = np.load('../assets/fountain/Rs/0004.npy')
    t1 = np.load('../assets/fountain/ts/0005.npy')
    t2 = np.load('../assets/fountain/ts/0004.npy')
    img1 = cv2.imread('../assets/fountain/images/0005.png')
    img2 = cv2.imread('../assets/fountain/images/0004.png')
    h, w, _ = img1.shape

    # resize the image to reduce computation
    scale = 8  # you could try different scale parameters, e.g. 4 for better quality & slower speed.
    img1 = cv2.resize(img1, (w // scale, h // scale))
    img2 = cv2.resize(img2, (w // scale, h // scale))
    h, w, _ = img1.shape

    # visualize the left and right image
    plt.figure()
    # opencv default color order is BGR instead of RGB, so we need to take care of it when visualization
    plt.imshow(cv2.cvtColor(np.concatenate((img1, img2), axis=1), cv2.COLOR_BGR2RGB))
    plt.title("Before rectification")
    plt.show()

    # Q6.a: How does intrinsic change before and after the scaling? You only need to modify K1 and K2 here,
    # if necessary. If you think they remain the same, leave here as blank and explain why.
    # --------------------------- Begin your code here ---------------------------------------------
    K1 /= scale
    K2 /= scale
    # --------------------------- End your code here   ---------------------------------------------

    # Compute the relative pose between two cameras
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3:] = t1
    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3:] = t2
    T = T2.dot(np.linalg.inv(T1))  # c1 to world and world to c2
    R = T[:3, :3]
    t = T[:3, 3:]

    # Rectify stereo image pair such that they are frontal parallel. Here we call cv2 to help us
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, None, K2, None, (w // 4, h // 4), R, t, 1,
                                                                      newImageSize=(0, 0))
    left_map = cv2.initUndistortRectifyMap(K1, None, R1, P1, (w, h), cv2.CV_16SC2)
    right_map = cv2.initUndistortRectifyMap(K2, None, R2, P2, (w, h), cv2.CV_16SC2)
    left_img = cv2.remap(img1, left_map[0], left_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    right_img = cv2.remap(img2, right_map[0], right_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    plt.figure()
    plt.imshow(cv2.cvtColor(np.concatenate((left_img, right_img), axis=1), cv2.COLOR_BGR2RGB))
    plt.title("After stereo rectification")
    plt.show()

    # Visualize images after rectification and report K1, K2 in your PDF report.

    disparity = stereo_matching_ssd(left_img, right_img, max_disp=128, block_size=7)
    # Depending on your implementation, runtime could be a few minutes. Feel free to try different hyper-parameters,
    # e.g. using a higher-resolution image, or a bigger block size. Do you see any difference?

    plt.figure()
    plt.imshow(disparity)
    plt.title("Disparity map")
    plt.show()

    # Compare your method and an off the shelf CV2's stereo matching results.
    # Please list a few directions which you think could improve your own results
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(np.concatenate((left_gray, right_gray), axis=1), 'gray')
    plt.title("Gray Scale Images")
    plt.show()

    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=7)
    disparity_cv2 = stereo.compute(left_gray, right_gray) / 16.0
    plt.imshow(np.concatenate((disparity, disparity_cv2), axis=1))
    plt.show()

    # Visualize disparity map and comparison against disparity_cv2 in your report.

    # Q6 Bonus:

    # --------------------------- Begin your code here ---------------------------------------------
    # Hints:
    # What is the focal length? How large is the stereo baseline?
    f = K1[0, 0]
    baseline = np.linalg.norm(T1 - T2)
    # Convert disparity to depth
    depth = f * baseline / (disparity + 1e-6)
    # Unproject image color and depth map to 3D point cloud
    cx, cy = K1[:2, -1]
    cx_prime = K1[0, -1]
    q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, f],
        [0, 0, -1 / baseline, (cx - cx_prime) / baseline]
    ])
    disparity_cv2 = stereo.compute(left_gray, right_gray)
    xyz = cv2.reprojectImageTo3D(disparity_cv2, q).reshape(-1, 3)
    xyz = xyz[~np.isinf(xyz).any(axis=1)]
    color = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB).reshape(-1, 3)  # / 255

    # depth = o3d.geometry.Image(np.clip(depth, 0, 255).astype(np.uint8))
    # left_img = o3d.geometry.Image(left_img)
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=left_img, depth=depth)
    # intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
    #     width=w, height=h, fx=K1[0][0], fy=K1[1][1], cx=K1[0][2], cy=K1[1][2]
    # )
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     image=rgbd, intrinsic=intrinsic_o3d, extrinsic=T1, project_valid_depth_only=True
    # )
    # You can use Open3D to visualize the colored point cloud
    # --------------------------- End your code here   ---------------------------------------------

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])
