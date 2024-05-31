import os
import numpy as np
import json
from pathlib import Path
import json
import cv2
import copy
import math
from src.edge_extraction.edge_fitting.main import edge_fit
from src.edge_extraction.merging.main import merge
from src.edge_extraction.extract_util import bezier_curve_length


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def get_edge_maps(data_dir, detector):
    meta = load_from_json(Path(data_dir) / "meta_data.json")
    h, w = meta["height"], meta["width"]
    edges_list, intrinsics_list, camtoworld_list = [], [], []
    for idx, frame in enumerate(meta["frames"]):
        intrinsics = np.array(frame["intrinsics"])
        camtoworld = np.array(frame["camtoworld"])[:4, :4]
        if detector == "DexiNed":
            edges_list.append(
                os.path.join(
                    data_dir,
                    "edge_DexiNed",
                    frame["rgb_path"],
                )
            )
        elif detector == "PidiNet":
            edges_list.append(
                os.path.join(
                    data_dir,
                    "edge_PidiNet",
                    frame["rgb_path"][:-4] + ".png",
                )
            )
        else:
            raise ValueError(f"Unknown detector: {detector}")
        intrinsics_list.append(intrinsics)
        camtoworld_list.append(camtoworld)

    edges = [cv2.imread(im_name, 0)[..., None] for im_name in edges_list]

    if detector == "DexiNed":
        edges = 1 - np.stack(edges) / 255.0
    elif detector == "PidiNet":
        edges = np.stack(edges) / 255.0

    intrinsics_list = np.stack(intrinsics_list)
    camtoworld_list = np.stack(camtoworld_list)
    return edges, intrinsics_list, camtoworld_list, h, w


def process_geometry_data(
    edge_dict,
    worldtogt=None,
    valid_curve=None,
    valid_line=None,
    sample_resolution=0.005,
):
    """
    Processes edge data to transform and sample points from geometric data (curves and lines).
    Optionally transforms points to a target coordinate system and filters specific geometries.

    Parameters:
        edge_dict (dict): Dictionary containing 'curves_ctl_pts' and 'lines_end_pts'.
        worldtogt (np.array, optional): Transformation matrix to convert points to another coordinate system.
        valid_curve (np.array, optional): Indices to filter specific curves.
        valid_line (np.array, optional): Indices to filter specific lines.
        sample_resolution (float): Sampling resolution for generating points.

    Returns:
        np.array: Array of sampled points.
        int: Number of curve points generated.
    """
    # Process curves
    return_edge_dict = {}
    curve_data = edge_dict["curves_ctl_pts"]
    curve_paras = np.array(curve_data).reshape(-1, 12)
    if valid_curve is not None:
        curve_paras = curve_paras[valid_curve]
    curve_paras = curve_paras.reshape(-1, 4, 3)
    return_edge_dict["curves_ctl_pts"] = curve_paras.tolist()

    if worldtogt is not None:
        curve_paras = curve_paras @ worldtogt[:3, :3].T + worldtogt[:3, 3]

    # Process lines
    line_data = edge_dict["lines_end_pts"]
    lines = np.array(line_data).reshape(-1, 6)
    if valid_line is not None:
        lines = lines[valid_line]

    return_edge_dict["lines_end_pts"] = lines.tolist()

    lines = lines.reshape(-1, 2, 3)

    if worldtogt is not None:
        lines = lines @ worldtogt[:3, :3].T + worldtogt[:3, 3]

    all_points = []

    # Sample curves
    for curve in curve_paras:
        sample_num = int(
            bezier_curve_length(curve, num_samples=100) // sample_resolution
        )
        t = np.linspace(0, 1, sample_num)
        coefficients = np.array(
            [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
        )
        matrix_u = np.array([t**3, t**2, t, np.ones_like(t)])
        points = matrix_u.T.dot(coefficients).dot(curve)
        all_points.extend(points.tolist())

    # Sample lines
    for line in lines:
        sample_num = int(np.linalg.norm(line[0] - line[1]) // sample_resolution)
        t = np.linspace(0, 1, sample_num)
        line_points = np.outer(t, line[1] - line[0]) + line[0]
        all_points.extend(line_points.tolist())

    return np.array(all_points, dtype=np.float32), return_edge_dict


def compute_visibility(
    all_curve_points,
    all_line_points,
    edges,
    intrinsics_list,
    camtoworld_list,
    h,
    w,
    edge_visibility_threshold,
    edge_visibility_frames,
):
    img_frames = len(edges)
    curve_num, line_num = len(all_curve_points), len(all_line_points)
    edge_num = curve_num + line_num
    edge_visibility_matrix = np.zeros((edge_num, img_frames))

    for frame_idx, (edge_map, intrinsic, camtoworld) in enumerate(
        zip(edges, intrinsics_list, camtoworld_list)
    ):
        K = intrinsic[:3, :3]
        worldtocam = np.linalg.inv(camtoworld)
        R = worldtocam[:3, :3]
        T = worldtocam[:3, 3:]

        all_curve_uv, all_line_uv = project2D(
            K, R, T, copy.deepcopy(all_curve_points), copy.deepcopy(all_line_points)
        )
        all_edge_uv = all_curve_uv + all_line_uv

        for edge_idx, edge_uv in enumerate(all_edge_uv):
            edge_uv = np.array(edge_uv)
            # print(edge_uv.shape)
            if len(edge_uv) == 0:
                continue
            edge_uv = np.round(edge_uv).astype(np.int32)
            edge_u = edge_uv[:, 0]
            edge_v = edge_uv[:, 1]

            valid_edge_uv = edge_uv[
                (edge_u >= 0) & (edge_u < w) & (edge_v >= 0) & (edge_v < h)
            ]
            visibility = 0

            if len(valid_edge_uv) > 0:
                projected_edge = edge_map[valid_edge_uv[:, 1], valid_edge_uv[:, 0]]
                visibility = float(
                    np.mean(projected_edge) > edge_visibility_threshold
                    and np.max(projected_edge) > 0.5
                )
                edge_visibility_matrix[edge_idx, frame_idx] = visibility

    return np.sum(edge_visibility_matrix, axis=1) > edge_visibility_frames


def project2D(K, R, T, all_curve_points, all_line_points):
    all_curve_uv, all_line_uv = [], []
    for curve_points in all_curve_points:
        curve_points = np.array(curve_points).reshape(-1, 3)
        curve_uv = project2D_single(K, R, T, curve_points)
        all_curve_uv.append(curve_uv)
    for line_points in all_line_points:
        line_points = np.array(line_points).reshape(-1, 3)
        line_uv = project2D_single(K, R, T, line_points)
        all_line_uv.append(line_uv)
    return all_curve_uv, all_line_uv


def project2D_single(K, R, T, points3d):
    shape = points3d.shape
    assert shape[-1] == 3
    X = points3d.reshape(-1, 3)

    x = K @ (R @ X.T + T)
    x = x.T
    x = x / x[:, -1:]
    x = x.reshape(*shape)[..., :2].reshape(-1, 2).tolist()
    return x


def get_parametric_edge(
    edge_dict,
    visible_checking=False,
):

    detector = edge_dict["detector"]
    scene_name = edge_dict["scene_name"]
    dataset_dir = edge_dict["dataset_dir"]
    result_dir = edge_dict["result_dir"]
    meta_data_dir = os.path.join(dataset_dir, scene_name)

    # fixed parameters, but can be fine-tuned for better edge extraction
    is_merge = True
    nms_factor = 0.95
    angle_threshold = 0.03
    fit_distance_threshold = 10.0
    min_inliers = 5
    max_lines = 4
    max_curves = 3
    merge_edge_distance_threshold = 5.0
    merge_endpoints_distance_threshold = 2.0
    merge_similarity_threshold = 0.98

    fitted_edge_dict = edge_fit(
        edge_data=edge_dict,
        angle_threshold=angle_threshold,
        nms_factor=nms_factor,
        fit_distance_threshold=fit_distance_threshold,
        min_inliers=min_inliers,
        max_lines=max_lines,
        max_curves=max_curves,
    )
    if is_merge:
        merged_edge_dict = merge(
            result_dir,
            fitted_edge_dict,
            merge_edge_distance_threshold=merge_edge_distance_threshold,
            merge_endpoints_distance_threshold=merge_endpoints_distance_threshold,
            merge_similarity_threshold=merge_similarity_threshold,
        )

    if visible_checking:
        _, return_edge_dict = process_geometry_data(merged_edge_dict)
        all_curve_points = return_edge_dict["curves_ctl_pts"]
        all_line_points = return_edge_dict["lines_end_pts"]
        edges, intrinsics_list, camtoworld_list, h, w = get_edge_maps(
            meta_data_dir, detector
        )
        edge_visibility_threshold = 0.5
        edge_visibility_frames_ratio = 0.1
        num_frames = len(edges)
        edge_visibility_frames = math.ceil(edge_visibility_frames_ratio * num_frames)

        edge_visibility = compute_visibility(
            all_curve_points,
            all_line_points,
            edges,
            intrinsics_list,
            camtoworld_list,
            h,
            w,
            edge_visibility_threshold,
            edge_visibility_frames,
        )
        curve_visibility = edge_visibility[: len(all_curve_points)]
        line_visibility = edge_visibility[len(all_curve_points) :]

        print(
            "before visible checking: ",
            len(all_curve_points) + len(all_line_points),
            "after visible checking: ",
            np.sum(edge_visibility),
        )
        worldtogt = np.eye(4)
        (pred_points, return_edge_dict) = process_geometry_data(
            merged_edge_dict, worldtogt, curve_visibility, line_visibility
        )

    else:
        worldtogt = np.eye(4)
        (pred_points, return_edge_dict) = process_geometry_data(
            merged_edge_dict, worldtogt, None, None
        )

    return pred_points, return_edge_dict
