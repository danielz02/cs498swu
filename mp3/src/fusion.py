import argparse
import os

import cv2
import volume
import numpy as np
import open3d as o3d
from icp import read_data, rgbd2pts
from volume import TSDFVolume

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
        depth_scale = 1
        voxel_size = 4.0 / 512
        trunc = voxel_size * 4
        device = o3d.core.Device("cpu:0")

        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
            attr_channels=(1, 1, 3),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000, device=device
        )

        color_im, depth_im, K, T_init = read_data(0)
        for i in range(step, n_imgs, step):
            print(f"Fusing frame {(i + 1)} / {n_imgs}")

            color_im, depth_im, K, T_src = read_data(i)
            color_im = o3d.t.geometry.Image(color_im)
            depth_im = o3d.t.geometry.Image(depth_im.astype(np.float32))

            K = o3d.core.Tensor(K)
            extrinsic = o3d.core.Tensor(np.linalg.inv(T_src) @ T_init)
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth_im, K, extrinsic, depth_scale=depth_scale, depth_max=depth_max
            )
            buf_indices, masks = vbg.hashmap().find(frustum_block_coords)
            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(buf_indices)

            extrinsic = extrinsic.to(device, o3d.core.float32)
            K = K.to(device, o3d.core.float32)
            xyz = extrinsic[:3, :3] @ voxel_coords.T() + extrinsic[:3, 3:]

            uvd = K @ xyz
            d = uvd[2]
            u = (uvd[0] / d).round().to(o3d.core.int64)
            v = (uvd[1] / d).round().to(o3d.core.int64)
            o3d.core.cuda.synchronize()

            mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth_im.columns) & (v < depth_im.rows)

            v_proj = v[mask_proj]
            u_proj = u[mask_proj]
            d_proj = d[mask_proj]

            depth_readings = depth_im.as_tensor()[v_proj, u_proj, 0].to(o3d.core.float32) / depth_scale
            sdf = depth_readings - d_proj

            mask_inlier = (depth_readings > 0) & (depth_readings < depth_max) & (sdf >= -trunc)

            sdf[sdf >= trunc] = trunc
            sdf = sdf / trunc
            weight = vbg.attribute('weight').reshape((-1, 1))
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))

            valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
            w = weight[valid_voxel_indices]
            wp = w + 1

            tsdf[valid_voxel_indices] \
                = (tsdf[valid_voxel_indices] * w +
                   sdf[mask_inlier].reshape(w.shape)) / wp
            color_readings = color_im.as_tensor()[v_proj, u_proj].to(o3d.core.float32)

            color = vbg.attribute('color').reshape((-1, 3))
            color[valid_voxel_indices] = (color[valid_voxel_indices] * w + color_readings[mask_inlier]) / wp

            weight[valid_voxel_indices] = wp
            o3d.core.cuda.synchronize()

        pcd = vbg.extract_point_cloud()
        mesh = vbg.extract_triangle_mesh()

        # verts, faces, norms, colors = tsdf_vol.get_mesh()
        # volume.meshwrite("mesh.ply", verts, faces, norms, colors)
        # mesh = o3d.io.read_triangle_mesh("mesh.ply")
        # os.remove("mesh.ply")

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh], zoom=0.422, front=[-0.11651295252277051, -0.047982289143896774, -0.99202945108647766],
        lookat=[0.023592929264511786, 0.051808635289583765, 1.7903649529102956],
        up=[0.097655832648056065, -0.9860023571949631, -0.13513952033284915]
    )
