import trimesh
import open3d as o3d
import numpy as np
import math
import json
import os
from src.eval.eval_util import (
    set_random_seeds,
    load_from_json,
    downsample_point_cloud_average,
)
from src.edge_extraction.edge_fitting.bezier_fit import bezier_curve
import argparse
from pathlib import Path
import json
import cv2


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def convert_ply_to_obj(ply_file_path, obj_file_path):
    # Load the .ply file
    mesh = trimesh.load(ply_file_path)
    # Export the mesh to .obj format
    mesh.export(obj_file_path, file_type="obj")


def save_point_cloud(file_path, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[: n1 + 1, : n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def convert_mesh_gt2world(mesh_path, out_mesh_path, gttoworld):
    mesh = trimesh.load(mesh_path)
    # mesh.transform(gttoworld)
    mesh.apply_transform(gttoworld)
    mesh.export(out_mesh_path, file_type="obj")
    return mesh


def get_edge_maps(data_dir):
    meta = load_from_json(Path(data_dir) / "meta_data.json")
    h, w = meta["height"], meta["width"]
    edges_list, intrinsics_list, camtoworld_list = [], [], []
    for idx, frame in enumerate(meta["frames"]):
        intrinsics = np.array(frame["intrinsics"])
        camtoworld = np.array(frame["camtoworld"])[:4, :4]
        edges_list.append(
            os.path.join(
                data_dir,
                "edge_PidiNet",
                frame["rgb_path"],
            )
        )
        intrinsics_list.append(intrinsics)
        camtoworld_list.append(camtoworld)

    edges = [cv2.imread(im_name, 0)[..., None] for im_name in edges_list]
    edges = 1 - np.stack(edges) / 255.0
    intrinsics_list = np.stack(intrinsics_list)
    camtoworld_list = np.stack(camtoworld_list)
    return edges, intrinsics_list, camtoworld_list, h, w


def compute_visibility(
    gt_points,
    edge_maps,
    intrinsics_list,
    camtoworld_list,
    h,
    w,
    edge_visibility_threshold,
    edge_visibility_frames,
):
    img_frames = len(edge_maps)
    point_visibility_matrix = np.zeros((len(gt_points), img_frames))

    for frame_idx, (edge_map, intrinsic, camtoworld) in enumerate(
        zip(edge_maps, intrinsics_list, camtoworld_list)
    ):
        K = intrinsic[:3, :3]
        worldtocam = np.linalg.inv(camtoworld)
        edge_uv = project2D(K, worldtocam, gt_points)
        edge_uv = np.round(edge_uv).astype(np.int64)

        # Boolean mask for valid u, v coordinates
        valid_u_mask = (edge_uv[:, 0] >= 0) & (edge_uv[:, 0] < w)
        valid_v_mask = (edge_uv[:, 1] >= 0) & (edge_uv[:, 1] < h)
        valid_mask = valid_u_mask & valid_v_mask

        valid_edge_uv = edge_uv[valid_mask]
        valid_projected_edges = edge_map[valid_edge_uv[:, 1], valid_edge_uv[:, 0]]

        # Calculate visibility in a vectorized manner
        visibility = (valid_projected_edges > edge_visibility_threshold).reshape(-1)

        point_visibility_matrix[valid_mask, frame_idx] = visibility.astype(float)

    return np.sum(point_visibility_matrix, axis=1) > edge_visibility_frames


def project2D(K, worldtocam, points3d):
    shape = points3d.shape
    R = worldtocam[:3, :3]
    T = worldtocam[:3, 3:]

    projected = K @ (R @ points3d.T + T)
    projected = projected.T
    projected = projected / projected[:, -1:]
    uv = projected.reshape(*shape)[..., :2].reshape(-1, 2)
    return uv


def bezier_curve_length(control_points, num_samples):
    def binomial_coefficient(n, i):
        return math.factorial(n) // (math.factorial(i) * math.factorial(n - i))

    def derivative_bezier(t):
        n = len(control_points) - 1
        point = np.array([0.0, 0.0, 0.0])
        for i, (p1, p2) in enumerate(zip(control_points[:-1], control_points[1:])):
            point += (
                n
                * binomial_coefficient(n - 1, i)
                * (1 - t) ** (n - 1 - i)
                * t**i
                * (np.array(p2) - np.array(p1))
            )
        return point

    def simpson_integral(a, b, num_samples):
        h = (b - a) / num_samples
        sum1 = sum(
            np.linalg.norm(derivative_bezier(a + i * h))
            for i in range(1, num_samples, 2)
        )
        sum2 = sum(
            np.linalg.norm(derivative_bezier(a + i * h))
            for i in range(2, num_samples - 1, 2)
        )
        return (
            (
                np.linalg.norm(derivative_bezier(a))
                + 4 * sum1
                + 2 * sum2
                + np.linalg.norm(derivative_bezier(b))
            )
            * h
            / 3
        )

    # Compute the length of the 3D Bezier curve using composite Simpson's rule
    length = 0.0
    for i in range(num_samples):
        t0 = i / num_samples
        t1 = (i + 1) / num_samples
        length += simpson_integral(t0, t1, num_samples)

    return length


def bezier_para_to_point_length(control_points, num_samples=100):
    t_fit = np.linspace(0, 1, num_samples)
    curve_point_set = []
    curve_length_set = []
    for control_point in control_points:
        points = bezier_curve(
            t_fit,
            control_point[0, 0],
            control_point[0, 1],
            control_point[0, 2],
            control_point[1, 0],
            control_point[1, 1],
            control_point[1, 2],
            control_point[2, 0],
            control_point[2, 1],
            control_point[2, 2],
            control_point[3, 0],
            control_point[3, 1],
            control_point[3, 2],
        )
        lengths = bezier_curve_length(control_point, num_samples=num_samples)
        curve_point_set.append(points)
        curve_length_set.append(lengths)
    return (
        np.array(curve_point_set).reshape(-1, num_samples, 3, 1),
        np.array(curve_length_set),
    )


def main(gt_point_cloud_dir, dataset_dir, out_dir):
    set_random_seeds()
    gt_point_cloud_dir = os.path.join(gt_point_cloud_dir, "Points", "stl")
    if not os.path.exists(gt_point_cloud_dir):
        print(
            f"Ground truth point cloud directory {gt_point_cloud_dir} does not exist. Please download it from http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
        )
        return

    scan_names_dict = {
        "scan37": [0.55, 0.3],
        "scan83": [0.65, 0.2],
        "scan105": [0.65, 0.2],
        "scan110": [0.5, 0.3],
        "scan118": [0.5, 0.3],
        "scan122": [0.35, 0.4],
    }

    os.makedirs(out_dir, exist_ok=True)

    for scan_name, (
        edge_visibility_threshold,
        edge_visibility_frames_ratio,
    ) in scan_names_dict.items():
        output_file = os.path.join(out_dir, scan_name, "edge_points.ply")
        if os.path.exists(output_file):
            print(f"{output_file} already exists. Skipping.")
            continue
        os.makedirs(os.path.join(out_dir, scan_name), exist_ok=True)
        meta_data_json_path = os.path.join(dataset_dir, scan_name, "meta_data.json")
        meta_base_dir = os.path.join(dataset_dir, scan_name)
        worldtogt = np.array(load_from_json(Path(meta_data_json_path))["worldtogt"])
        gttoworld = np.linalg.inv(worldtogt)
        gt_point_cloud_path = os.path.join(
            gt_point_cloud_dir,
            f"stl{int(scan_name[4:]):03d}_total.ply",
        )
        gt_point_cloud = o3d.io.read_point_cloud(gt_point_cloud_path)

        gt_points = np.asarray(gt_point_cloud.points)
        points = gt_points @ gttoworld[:3, :3].T + gttoworld[:3, 3][None, ...]

        edge_maps, intrinsics_list, camtoworld_list, h, w = get_edge_maps(meta_base_dir)
        num_frames = len(edge_maps)
        edge_visibility_frames = max(
            1, round(edge_visibility_frames_ratio * num_frames)
        )
        points_visibility = compute_visibility(
            points,
            edge_maps,
            intrinsics_list,
            camtoworld_list,
            h,
            w,
            edge_visibility_threshold,
            edge_visibility_frames,
        )

        print(
            f"{scan_name}: before visibility check: {len(points)}, after visibility check: {np.sum(points_visibility)}"
        )

        edge_points = points[points_visibility]
        downsampled_edge_points = downsample_point_cloud_average(
            edge_points, num_voxels_per_axis=256
        )
        downsampled_edge_points = (
            downsampled_edge_points @ worldtogt[:3, :3].T + worldtogt[:3, 3][None, ...]
        )
        save_point_cloud(output_file, downsampled_edge_points)
        print(f"Saved downsampled edge point cloud to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and evaluate point cloud data."
    )
    parser.add_argument(
        "--gt_point_cloud_dir",
        type=str,
        default="data/DTU_Edge/groundtruth",
    )
    parser.add_argument("--dataset_dir", type=str, default="data/DTU_Edge/data")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/DTU_Edge/groundtruth/edge_points",
    )
    args = parser.parse_args()

    main(args.gt_point_cloud_dir, args.dataset_dir, args.out_dir)
