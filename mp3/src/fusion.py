import argparse
import numpy as np
import open3d as o3d
from glob import glob
from icp import read_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    n_imgs = 100
    step = args.step
    if args.reference:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        color_im, depth_im, K, T_init = read_data(0)
        for i in range(step, n_imgs, step):
            color_im, depth_im, K, T_src = read_data(i)

            print("Integrate {:d}-th image into the volume.".format(i))
            color = o3d.io.read_image(f"../data/frame-{str(i).zfill(6)}.color.jpg")
            depth = o3d.io.read_image(f"../data/frame-{str(i).zfill(6)}.depth.png")
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
            T_W2C_gt = np.linalg.inv(T_src) @ T_init
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    depth_im.shape[1], depth_im.shape[0], K[0, 0], K[1, 1], K[0, -1], K[1, -1]
                ),
                T_W2C_gt
            )

        mesh = volume.extract_triangle_mesh()
    else:
        # Reference 1: https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py
        # Reference 2: http://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/customized_integration.html
        # Reference 3: https://github.com/chengkunli96/KinectFusion/tree/main

        res = 8
        depth_max = 3.0
        depth_scale = 1000
        voxel_size = 3.0 / 512
        trunc = voxel_size * 4
        device = o3d.core.Device("CUDA:0")

        depth_img = sorted(glob("../data/*.depth.png"))
        color_img = sorted(glob("../data/*.color.jpg"))
        intrinsic = o3d.core.Tensor(np.loadtxt("../data/camera-intrinsics.txt", delimiter=' '))
        t_init = np.loadtxt("../data/frame-000000.pose.txt")
        extrinsics = [np.linalg.inv(np.loadtxt("../data/frame-%06d.pose.txt" % ind)) @ t_init for ind in range(100)]

        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
            attr_channels=(1, 1, 3),
            voxel_size=voxel_size,
            block_resolution=8,
            block_count=100000, device=device
        )

        for i in range(0, n_imgs, step):
            print(f"Fusing frame {(i + 1)} / {n_imgs}")

            depth = o3d.t.io.read_image(depth_img[i]).to(device)
            extrinsic = o3d.core.Tensor(extrinsics[i])

            # Get active frustum block coordinates from input
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, depth_scale, depth_max
            )
            # Activate them in the underlying hash map (may have been inserted)
            vbg.hashmap().activate(frustum_block_coords)

            # Find buf indices in the underlying engine
            buf_indices, masks = vbg.hashmap().find(frustum_block_coords)
            o3d.core.cuda.synchronize()

            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(buf_indices)
            o3d.core.cuda.synchronize()

            # Voxel to frame coordinate
            extrinsic = extrinsic.to(device, o3d.core.float32)
            xyz = extrinsic[:3, :3] @ voxel_coords.T() + extrinsic[:3, 3:]

            intrinsic_dev = intrinsic.to(device, o3d.core.float32)
            uvd = intrinsic_dev @ xyz
            d = uvd[2]
            u = (uvd[0] / d).round().to(o3d.core.int64)
            v = (uvd[1] / d).round().to(o3d.core.int64)
            o3d.core.cuda.synchronize()

            # Remove the points that go outside the frame
            mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (v < depth.rows)

            # Calculate the SDF for those points
            v_proj = v[mask_proj]
            u_proj = u[mask_proj]
            d_proj = d[mask_proj]
            depth_readings = depth.as_tensor()[v_proj, u_proj, 0].to(o3d.core.float32) / depth_scale
            tsdf_new = depth_readings - d_proj

            # Check inliers based on depth information
            mask_inlier = (depth_readings > 0) & (depth_readings < depth_max) & (tsdf_new >= -trunc)

            # Truncate SDF
            tsdf_new[tsdf_new >= trunc] = trunc
            tsdf_new = tsdf_new / trunc
            o3d.core.cuda.synchronize()

            # Get old weight and TSDF
            weight = vbg.attribute('weight').reshape((-1, 1))
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))

            # Weight update
            valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
            w = weight[valid_voxel_indices]
            wp = w + 1

            # TSDF update
            tsdf[valid_voxel_indices] = (tsdf[valid_voxel_indices] * w + tsdf_new[mask_inlier].reshape(w.shape)) / wp
            color = o3d.t.io.read_image(color_img[i]).to(device)
            color_readings = color.as_tensor()[v_proj, u_proj].to(o3d.core.float32)

            # Color information update
            color = vbg.attribute('color').reshape((-1, 3))
            color[valid_voxel_indices] = (color[valid_voxel_indices] * w + color_readings[mask_inlier]) / wp

            # Save and synchronize new weights
            weight[valid_voxel_indices] = wp
            o3d.core.cuda.synchronize()

        mesh = vbg.extract_triangle_mesh().to_legacy()

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh], zoom=0.422, front=[-0.11651295252277051, -0.047982289143896774, -0.99202945108647766],
        lookat=[0.023592929264511786, 0.051808635289583765, 1.7903649529102956],
        up=[0.097655832648056065, -0.9860023571949631, -0.13513952033284915]
    )
