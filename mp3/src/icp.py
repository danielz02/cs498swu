from re import I
import time
import cv2
import numpy as np
import open3d as o3d
import time
from sklearn.neighbors import KDTree


# Question 4: deal with point_to_plane = True
def fit_rigid(src, tgt, tgt_normals=None, point_to_plane=False):
    # Question 2: Rigid Transform Fitting
    # Implement this function
    # -------------------------
    if point_to_plane:
        # Reference: https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
        assert tgt_normals is not None
        c = np.cross(src, tgt_normals)
        cn = np.concatenate([c, tgt_normals], axis=-1)
        a = (cn[..., np.newaxis] @ cn[:, np.newaxis, :]).sum(axis=0)
        b = -np.sum(np.sum((src - tgt) * tgt_normals, axis=1).reshape(-1, 1) * cn, axis=0)
        a_inv = np.linalg.pinv(a)
        alpha, beta, gamma, tx, ty, tz = a_inv @ b
        pose = np.array([
            [1, -gamma, beta, tx],
            [gamma, 1, -alpha, ty],
            [-beta, alpha, 1, tz],
            [0, 0, 0, 1]
        ])
    else:
        p_bar = src.mean(axis=0)
        q_bar = tgt.mean(axis=0)

        u, _, vh = np.linalg.svd((tgt - q_bar).T @ (src - p_bar))
        r = u @ vh

        t = q_bar - r @ p_bar

        pose = np.eye(4)
        pose[:3, :3] = r
        pose[:-1, -1] = t

    # -------------------------
    return pose


# Question 4: deal with point_to_plane = True
def icp(src, tgt, init_pose=np.eye(4), max_iter=20, point_to_plane=False):
    if point_to_plane:
        tgt_normals = np.asarray(tgt.normals)
    else:
        src_normals, tgt_normals = None, None

    src = np.asarray(src.points)
    tgt = np.asarray(tgt.points)

    # Question 3: ICP
    # Hint 1: using KDTree for fast nearest neighbour
    # Hint 3: you should be calling fit_rigid inside the loop
    # Your implementation between the lines
    # ---------------------------------------------------
    t = init_pose
    transforms = []
    delta_ts = []

    inlier_ratio = 0
    print("iter %d: inlier ratio: %.2f" % (0, inlier_ratio))

    for k in range(max_iter):
        # ---------------------------------------------------
        src_proj = (t[:3, :3] @ src.T).T + t[:3, -1].reshape(-1, 3)
        kd = KDTree(tgt)
        dist, idx_closest = kd.query(src_proj, k=1, return_distance=True)
        dist = np.array(dist).reshape(-1)
        idx_closest = np.array(idx_closest).reshape(-1)
        normals = tgt_normals[idx_closest] if tgt_normals is not None else None
        t_delta = fit_rigid(src_proj, tgt[idx_closest], normals, point_to_plane)
        t = t_delta @ t

        print("iter %d: inlier ratio: %.2f" % (k + 1, inlier_ratio))
        # relative update from each iteration
        delta_ts.append(t_delta.copy())
        # pose estimation after each iteration
        transforms.append(t.copy())

        inlier_ratio = (dist < 0.1).mean()
        if inlier_ratio > 0.999:
            break

    return transforms, delta_ts


def rgbd2pts(color, depth, k):
    # Question 1: unproject rgbd to color point cloud, provide visualization in your document
    # Your implementation between the lines
    # ---------------------------

    h, w = depth.shape
    n = h * w
    depth = depth.reshape(1, -1)  # (1, n)
    color = color.reshape(-1, 3)
    coords = np.vstack([np.stack(np.meshgrid(range(w), range(h)), axis=0).reshape(2, -1), np.ones(n).reshape(1, -1)])
    xyz = (np.linalg.inv(k) @ coords) * depth  # (1, n) * (3, 3) @ (3, n)
    # ---------------------------

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.T)
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


# TODO (Shenlong): please check that I set this question up correctly, it is called on line 136
def pose_error(estimated_pose, gt):
    # Question 5: Translation and Rotation Error
    # Use equations 5-6 in https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf
    # Your implementation between the lines
    # ---------------------------
    r_hat, r_gt = estimated_pose[:3, :3], gt[:3, :3]
    t_hat, t_gt = estimated_pose[:-1, -1], gt[:-1, -1]
    te = np.linalg.norm(t_hat - t_gt)
    re = np.arccos((np.trace(r_hat @ np.linalg.inv(r_gt)) - 1) / 2)
    # ---------------------------
    return re, te


def read_data(ind=0):
    k = np.loadtxt("../data/camera-intrinsics.txt", delimiter=' ')
    depth_img = cv2.imread("../data/frame-%06d.depth.png" % ind, -1).astype(float)
    depth_img /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_img[depth_img == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    t = np.loadtxt("../data/frame-%06d.pose.txt" % ind)  # 4x4 rigid transformation matrix
    color_img = cv2.imread("../data/frame-%06d.color.jpg" % ind, -1)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) / 255.0
    return color_img, depth_img, k, t


if __name__ == "__main__":
    # pairwise ICP
    # read color, image data and the ground-truth, converting to point cloud
    color_im, depth_im, K, T_tgt = read_data(0)
    target = rgbd2pts(color_im, depth_im, K)
    color_im, depth_im, K, T_src = read_data(40)
    source = rgbd2pts(color_im, depth_im, K)

    # downsampling and normal estimation
    source = source.voxel_down_sample(voxel_size=0.02)
    target = target.voxel_down_sample(voxel_size=0.02)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # conduct ICP (your code)
    final_Ts, delta_Ts = icp(source, target)

    # visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    ctr.set_front([-0.11651295252277051, -0.047982289143896774, -0.99202945108647766])
    ctr.set_lookat([0.023592929264511786, 0.051808635289583765, 1.7903649529102956])
    ctr.set_up([0.097655832648056065, -0.9860023571949631, -0.13513952033284915])
    ctr.set_zoom(0.42199999999999971)
    vis.add_geometry(source)
    vis.add_geometry(target)

    save_image = False

    # update source images
    for i in range(len(delta_Ts)):
        source.transform(delta_Ts[i])
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.2)
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)

    # visualize camera
    h_im, w_im, _ = color_im.shape
    tgt_cam = o3d.geometry.LineSet.create_camera_visualization(w_im, h_im, K, np.eye(4), scale=0.2)
    src_cam = o3d.geometry.LineSet.create_camera_visualization(w_im, h_im, K, np.linalg.inv(T_src) @ T_tgt, scale=0.2)
    pred_cam = o3d.geometry.LineSet.create_camera_visualization(w_im, h_im, K, np.linalg.inv(final_Ts[-1]), scale=0.2)

    gt_pose = np.linalg.inv(T_src) @ T_tgt
    pred_pose = np.linalg.inv(final_Ts[-1])
    p_error = pose_error(pred_pose, gt_pose)
    print("Ground truth pose:", gt_pose)
    print("Estimated pose:", pred_pose)
    print("Rotation/Translation Error", p_error)

    tgt_cam.paint_uniform_color((1, 0, 0))
    src_cam.paint_uniform_color((0, 1, 0))
    pred_cam.paint_uniform_color((0, 0.5, 0.5))
    vis.add_geometry(src_cam)
    vis.add_geometry(tgt_cam)
    vis.add_geometry(pred_cam)

    vis.run()
    vis.destroy_window()

    # Provide visualization of alignment with camera poses in write-up.
    # Print pred pose vs gt pose in write-up.
