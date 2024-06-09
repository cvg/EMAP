import os
import numpy as np
import json
import argparse
import random
from pathlib import Path
from src.eval.eval_util import (
    compute_chamfer_distance,
    f_score,
    compute_precision_recall_IOU,
    downsample_point_cloud_average,
    get_gt_points,
    get_pred_points_and_directions,
)


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename."""
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def update_totals_and_metrics(metrics, totals, results, edge_type):
    correct_gt, num_gt, correct_pred, num_pred, acc, comp = results
    metrics[f"comp_{edge_type}"].append(comp)
    metrics[f"acc_{edge_type}"].append(acc)
    for i, threshold in enumerate(["5", "10", "20"]):
        totals[f"thre{threshold}_correct_gt_total"] += correct_gt[i]
        totals[f"thre{threshold}_correct_pred_total"] += correct_pred[i]
    totals["num_gt_total"] += num_gt
    totals["num_pred_total"] += num_pred


def finalize_metrics(metrics):
    for key, value in metrics.items():
        value = np.array(value)
        value[np.isnan(value)] = 0
        metrics[key] = round(np.mean(value), 4)
    return metrics


def print_metrics(metrics, totals, edge_type):
    print(f"{edge_type.capitalize()}:")
    print(f"  Completeness: {metrics[f'comp_{edge_type}']}")
    print(f"  Accuracy: {metrics[f'acc_{edge_type}']}")


def process_scan(scan_name, base_dir, exp_name, dataset_dir, metrics, totals):
    print(f"Processing: {scan_name}")
    json_path = os.path.join(
        base_dir, scan_name, exp_name, "results", "parametric_edges.json"
    )
    if not os.path.exists(json_path):
        print(f"Invalid prediction at {scan_name}")
        return

    all_curve_points, all_line_points, all_curve_directions, all_line_directions = (
        get_pred_points_and_directions(json_path)
    )
    pred_points = (
        np.concatenate([all_curve_points, all_line_points], axis=0)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    if len(pred_points) == 0:
        print(f"Invalid prediction at {scan_name}")
        return

    pred_sampled = downsample_point_cloud_average(
        pred_points,
        num_voxels_per_axis=256,
        min_bound=[-1, -1, -1],
        max_bound=[1, 1, 1],
    )

    gt_points_raw, gt_points, _ = get_gt_points(
        scan_name, "all", data_base_dir=os.path.join(dataset_dir, "groundtruth")
    )
    if gt_points_raw is None:
        return

    chamfer_dist, acc, comp = compute_chamfer_distance(pred_sampled, gt_points)
    print(
        f"  Chamfer Distance: {chamfer_dist:.4f}, Accuracy: {acc:.4f}, Completeness: {comp:.4f}"
    )
    metrics["chamfer"].append(chamfer_dist)
    metrics["acc"].append(acc)
    metrics["comp"].append(comp)
    metrics = compute_precision_recall_IOU(
        pred_sampled,
        gt_points,
        metrics,
        thresh_list=[0.005, 0.01, 0.02],
        edge_type="all",
    )

    for edge_type in ["curve", "line"]:
        gt_points_raw_edge, gt_points_edge, _ = get_gt_points(
            scan_name,
            edge_type,
            return_direction=True,
            data_base_dir=os.path.join(dataset_dir, "groundtruth"),
        )
        if gt_points_raw_edge is not None:
            results = compute_precision_recall_IOU(
                pred_sampled,
                gt_points_edge,
                None,
                thresh_list=[0.005, 0.01, 0.02],
                edge_type=edge_type,
            )
            update_totals_and_metrics(metrics, totals[edge_type], results, edge_type)


def main(base_dir, dataset_dir, exp_name):
    set_random_seeds()
    metrics = {
        "chamfer": [],
        "acc": [],
        "comp": [],
        "comp_curve": [],
        "comp_line": [],
        "acc_curve": [],
        "acc_line": [],
        "precision_0.01": [],
        "recall_0.01": [],
        "fscore_0.01": [],
        "IOU_0.01": [],
        "precision_0.02": [],
        "recall_0.02": [],
        "fscore_0.02": [],
        "IOU_0.02": [],
        "precision_0.005": [],
        "recall_0.005": [],
        "fscore_0.005": [],
        "IOU_0.005": [],
    }

    totals = {
        "curve": {
            "thre5_correct_gt_total": 0,
            "thre10_correct_gt_total": 0,
            "thre20_correct_gt_total": 0,
            "thre5_correct_pred_total": 0,
            "thre10_correct_pred_total": 0,
            "thre20_correct_pred_total": 0,
            "num_gt_total": 0,
            "num_pred_total": 0,
        },
        "line": {
            "thre5_correct_gt_total": 0,
            "thre10_correct_gt_total": 0,
            "thre20_correct_gt_total": 0,
            "thre5_correct_pred_total": 0,
            "thre10_correct_pred_total": 0,
            "thre20_correct_pred_total": 0,
            "num_gt_total": 0,
            "num_pred_total": 0,
        },
    }

    with open("src/eval/eval_scans.txt", "r") as f:
        scan_names = [line.strip() for line in f]

    for scan_name in scan_names:
        process_scan(scan_name, base_dir, exp_name, dataset_dir, metrics, totals)

    metrics = finalize_metrics(metrics)

    print("Summary:")
    print(f"  Accuracy: {metrics['acc']:.4f}")
    print(f"  Completeness: {metrics['comp']:.4f}")
    print(f"  Recall @ 5 mm: {metrics['recall_0.005']:.4f}")
    print(f"  Recall @ 10 mm: {metrics['recall_0.01']:.4f}")
    print(f"  Recall @ 20 mm: {metrics['recall_0.02']:.4f}")
    print(f"  Precision @ 5 mm: {metrics['precision_0.005']:.4f}")
    print(f"  Precision @ 10 mm: {metrics['precision_0.01']:.4f}")
    print(f"  Precision @ 20 mm: {metrics['precision_0.02']:.4f}")
    print(f"  F-Score @ 5 mm: {metrics['fscore_0.005']:.4f}")
    print(f"  F-Score @ 10 mm: {metrics['fscore_0.01']:.4f}")
    print(f"  F-Score @ 20 mm: {metrics['fscore_0.02']:.4f}")

    if totals["curve"]["num_gt_total"] > 0:
        print_metrics(metrics, totals["curve"], "curve")
    else:
        print("Curve: No ground truth edges found.")

    if totals["line"]["num_gt_total"] > 0:
        print_metrics(metrics, totals["line"], "line")
    else:
        print("Line: No ground truth edges found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CAD data and compute metrics."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./exp/ABC",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/ABC-NEF_Edge",
        help="Directory for the dataset",
    )
    parser.add_argument("--exp_name", type=str, default="emap", help="Experiment name")

    args = parser.parse_args()
    main(args.base_dir, args.dataset_dir, args.exp_name)
