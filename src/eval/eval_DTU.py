import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os
from pathlib import Path
from src.eval.eval_util import (
    set_random_seeds,
    load_from_json,
    downsample_point_cloud_average,
    get_pred_points_and_directions,
)


def process_scan(
    scan_name,
    base_dir,
    exp_name,
    dataset_dir,
    threshold,
    downsample_density,
    precision_list,
    recall_list,
):
    print(f"Processing: {scan_name}")
    json_path = os.path.join(
        base_dir, scan_name, exp_name, "results", "parametric_edges.json"
    )
    if not os.path.exists(json_path):
        print(f"Invalid prediction at {scan_name}")
        return

    meta_data_json_path = os.path.join(dataset_dir, "data", scan_name, "meta_data.json")
    worldtogt = np.array(load_from_json(Path(meta_data_json_path))["worldtogt"])

    all_curve_points, all_line_points, _, _ = get_pred_points_and_directions(json_path)
    all_points = np.concatenate([all_curve_points, all_line_points], axis=0).reshape(
        -1, 3
    )
    all_points = np.dot(all_points, worldtogt[:3, :3].T) + worldtogt[:3, 3]

    points_down = downsample_point_cloud_average(all_points, num_voxels_per_axis=256)

    nn_engine = skln.NearestNeighbors(
        n_neighbors=1, radius=downsample_density, algorithm="kd_tree", n_jobs=-1
    )

    gt_edge_points_path = os.path.join(
        dataset_dir, "groundtruth", "edge_points", scan_name, "edge_points.ply"
    )
    gt_edge_pcd = o3d.io.read_point_cloud(gt_edge_points_path)
    gt_edge_points = np.asarray(gt_edge_pcd.points)

    nn_engine.fit(gt_edge_points)
    dist_d2s, idx_d2s = nn_engine.kneighbors(
        points_down, n_neighbors=1, return_distance=True
    )
    precision = np.sum(dist_d2s <= threshold) / dist_d2s.shape[0]
    precision_list.append(precision)

    nn_engine.fit(points_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(
        gt_edge_points, n_neighbors=1, return_distance=True
    )
    recall = np.sum(dist_s2d <= threshold) / len(dist_s2d)
    recall_list.append(recall)

    print(f"  Recall: {recall:.4f}, Precision: {precision:.4f}")


def main(args):
    set_random_seeds()
    with open("src/eval/DTU_scans.txt", "r") as f:
        scan_names = [line.strip() for line in f]

    precision_list = []
    recall_list = []

    for scan_name in scan_names:
        process_scan(
            scan_name,
            args.base_dir,
            args.exp_name,
            args.dataset_dir,
            args.threshold,
            args.downsample_density,
            precision_list,
            recall_list,
        )

    print("\nSummary:")
    print(f"  Mean Recall: {np.mean(recall_list):.4f}")
    print(f"  Mean Precision: {np.mean(precision_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DTU data and compute metrics."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./exp/DTU",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/DTU_Edge",
        help="Directory for the dataset",
    )
    parser.add_argument("--exp_name", type=str, default="emap", help="Experiment name")
    parser.add_argument("--downsample_density", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=5)
    args = parser.parse_args()
    main(args)
