import os
import numpy as np
import json
import argparse
from pathlib import Path
import json
import random
from src.eval.eval_util import (
    compute_chamfer_distance,
    f_score,
    compute_precision_recall_IOU,
    downsample_point_cloud_average,
    get_gt_points,
    get_pred_points_and_directions,
)


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def set_random_seeds(seed=42):
    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for the random module
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CAD data and compute metrics."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./exp/ABC",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/ABC-NEF_Edge",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="emap",
    )

    args = parser.parse_args()
    base_dir = args.base_dir
    dataset_dir = args.dataset_dir
    exp_name = args.exp_name

    set_random_seeds()
    metrics_total = {}

    with open("eval/eval_scans.txt", "r") as f:
        scan_names = f.readlines()

    scan_names = [each.replace("\n", "") for each in scan_names]
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

    (
        thre5_correct_gt_total_curve,
        thre10_correct_gt_total_curve,
        thre20_correct_gt_total_curve,
        thre5_correct_pred_total_curve,
        thre10_correct_pred_total_curve,
        thre20_correct_pred_total_curve,
    ) = (0, 0, 0, 0, 0, 0)
    num_gt_total_curve = 0
    num_pred_total_curve = 0

    (
        thre5_correct_gt_total_line,
        thre10_correct_gt_total_line,
        thre20_correct_gt_total_line,
        thre5_correct_pred_total_line,
        thre10_correct_pred_total_line,
        thre20_correct_pred_total_line,
    ) = (0, 0, 0, 0, 0, 0)
    num_gt_total_line = 0
    num_pred_total_line = 0
    line_normal_similarities_list = []
    curve_normal_similarities_list = []

    for i, scan_name in enumerate(scan_names):
        print("Processing:", scan_name)
        json_path = os.path.join(
            base_dir,
            scan_name,
            exp_name,
            "results",
            "parametric_edges.json",
        )
        if not os.path.exists(json_path):
            print("invalid prediction at {}".format(scan_name))
            continue

        all_curve_points, all_line_points, all_curve_directions, all_line_directions = (
            get_pred_points_and_directions(json_path)
        )
        # pred_points = np.array(pred_points).reshape(-1, 3)
        pred_points = (
            np.concatenate([all_curve_points, all_line_points], axis=0)
            .reshape(-1, 3)
            .astype(np.float32)
        )

        if len(pred_points) == 0:
            print("invalid prediction at {}".format(scan_name))
            continue

        pred_sampled = downsample_point_cloud_average(
            pred_points,
            num_voxels_per_axis=256,
            min_bound=[-1, -1, -1],
            max_bound=[1, 1, 1],
        )

        #### all edges
        gt_points_raw, gt_points, _ = get_gt_points(
            scan_name, "all", data_base_dir=os.path.join(dataset_dir, "groundtruth")
        )
        if gt_points_raw is None:
            continue
        chamfer_dist, acc, comp = compute_chamfer_distance(pred_sampled, gt_points)
        print("chamfer:", chamfer_dist, "acc:", acc, "comp:", comp)
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

        #### curve and line
        gt_points_raw_curve, gt_points_curve, gt_curve_normals = get_gt_points(
            scan_name,
            "curve",
            return_direction=True,
            data_base_dir=os.path.join(dataset_dir, "groundtruth"),
        )
        if gt_points_raw_curve is not None:
            (
                correct_gt_curve,
                num_gt_curve,
                correct_pred_curve,
                num_pred_curve,
                acc_curve,
                comp_curve,
            ) = compute_precision_recall_IOU(
                pred_sampled,
                gt_points_curve,
                None,
                thresh_list=[0.005, 0.01, 0.02],
                edge_type="curve",
            )
            metrics["comp_curve"].append(comp_curve)
            metrics["acc_curve"].append(acc_curve)
            thre5_correct_gt_total_curve += correct_gt_curve[0]
            thre10_correct_gt_total_curve += correct_gt_curve[1]
            thre20_correct_gt_total_curve += correct_gt_curve[2]
            num_gt_total_curve += num_gt_curve

            thre5_correct_pred_total_curve += correct_pred_curve[0]
            thre10_correct_pred_total_curve += correct_pred_curve[1]
            thre20_correct_pred_total_curve += correct_pred_curve[2]
            num_pred_total_curve += num_pred_curve

            # curve_normal_similarity = compute_direction_similarity(
            #     gt_points_curve,
            #     gt_curve_normals,
            #     pred_sampled,
            #     directions,
            # )
            # print("curve normal similarity:", curve_normal_similarity)
            # curve_normal_similarities_list.append(curve_normal_similarity)

        gt_points_raw_line, gt_points_line, gt_line_normals = get_gt_points(
            scan_name,
            "line",
            return_direction=True,
            data_base_dir=os.path.join(dataset_dir, "groundtruth"),
        )
        if gt_points_raw_line is not None:
            (
                correct_gt_line,
                num_gt_line,
                correct_pred_line,
                num_pred_line,
                acc_line,
                comp_line,
            ) = compute_precision_recall_IOU(
                pred_sampled,
                gt_points_line,
                None,
                thresh_list=[0.005, 0.01, 0.02],
                edge_type="line",
            )
            metrics["comp_line"].append(comp_line)
            metrics["acc_line"].append(acc_line)
            thre5_correct_gt_total_line += correct_gt_line[0]
            thre10_correct_gt_total_line += correct_gt_line[1]
            thre20_correct_gt_total_line += correct_gt_line[2]
            num_gt_total_line += num_gt_line

            thre5_correct_pred_total_line += correct_pred_line[0]
            thre10_correct_pred_total_line += correct_pred_line[1]
            thre20_correct_pred_total_line += correct_pred_line[2]
            num_pred_total_line += num_pred_line

            # line_normal_similarity = compute_direction_similarity(
            #     gt_points_line,
            #     gt_line_normals,
            #     pred_sampled,
            #     directions,
            # )
            # print("line normal similarity:", line_normal_similarity)
            # line_normal_similarities_list.append(line_normal_similarity)

    for key, value in metrics.items():
        value = np.array(value)
        value[np.isnan(value)] = 0
        metrics[key] = round(np.mean(value), 4)

    print("All:")
    print("acc: ", metrics["acc"], "comp: ", metrics["comp"])
    print(
        "r5: ",
        metrics["recall_0.005"],
        "r10: ",
        metrics["recall_0.01"],
        "r20: ",
        metrics["recall_0.02"],
    )
    print(
        "p5: ",
        metrics["precision_0.005"],
        "p10: ",
        metrics["precision_0.01"],
        "p20: ",
        metrics["precision_0.02"],
    )
    print(
        "f5: ",
        metrics["fscore_0.005"],
        "f10: ",
        metrics["fscore_0.01"],
        "f20: ",
        metrics["fscore_0.02"],
    )

    print("Curve:")

    print("recall_0.005: ", 1.0 * thre5_correct_gt_total_curve / num_gt_total_curve)
    print("recall_0.01: ", 1.0 * thre10_correct_gt_total_curve / num_gt_total_curve)
    print("recall_0.02: ", 1.0 * thre20_correct_gt_total_curve / num_gt_total_curve)
    print("Curve completeness: ", metrics["comp_curve"])
    print(
        "precision_0.005: ", 1.0 * thre5_correct_pred_total_curve / num_pred_total_curve
    )
    print(
        "precision_0.01: ", 1.0 * thre10_correct_pred_total_curve / num_pred_total_curve
    )
    print(
        "precision_0.02: ", 1.0 * thre20_correct_pred_total_curve / num_pred_total_curve
    )
    print("Curve accuracy: ", metrics["acc_curve"])
    print(
        "f-score_0.005: ",
        f_score(
            1.0 * thre5_correct_pred_total_curve / num_pred_total_curve,
            1.0 * thre5_correct_gt_total_curve / num_gt_total_curve,
        ),
    )

    print("Line:")
    print("recall_0.005: ", 1.0 * thre5_correct_gt_total_line / num_gt_total_line)
    print("recall_0.01: ", 1.0 * thre10_correct_gt_total_line / num_gt_total_line)
    print("recall_0.02: ", 1.0 * thre20_correct_gt_total_line / num_gt_total_line)
    print("Line completeness: ", metrics["comp_line"])
    print(
        "precision_0.005: ", 1.0 * thre5_correct_pred_total_line / num_pred_total_line
    )
    print(
        "precision_0.01: ", 1.0 * thre10_correct_pred_total_line / num_pred_total_line
    )
    print(
        "precision_0.02: ", 1.0 * thre20_correct_pred_total_line / num_pred_total_line
    )
    print("Line accuracy: ", metrics["acc_line"])
    print(
        "f-score_0.005: ",
        f_score(
            1.0 * thre5_correct_pred_total_line / num_pred_total_line,
            1.0 * thre5_correct_gt_total_line / num_gt_total_line,
        ),
    )
