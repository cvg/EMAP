import os
import numpy as np
import json
import point_cloud_utils as pcu
import math


def chamfer_distance(x, y, return_index=False, p_norm=2, max_points_per_leaf=10):
    """
    Compute the chamfer distance between two point clouds x, and y

    Args:
        x : A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
        y : A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
        return_index: If set to True, will return a pair (corrs_x_to_y, corrs_y_to_x) where
                    corrs_x_to_y[i] stores the index into y of the closest point to x[i]
                    (i.e. y[corrs_x_to_y[i]] is the nearest neighbor to x[i] in y).
                    corrs_y_to_x is similar to corrs_x_to_y but with x and y reversed.
        max_points_per_leaf : The maximum number of points per leaf node in the KD tree used by this function.
                            Default is 10.
        p_norm : Which norm to use. p_norm can be any real number, inf (for the max norm) -inf (for the min norm),
                0 (for sum(x != 0))
    Returns:
        The chamfer distance between x an dy.
        If return_index is set, then this function returns a tuple (chamfer_dist, corrs_x_to_y, corrs_y_to_x) where
        corrs_x_to_y and corrs_y_to_x are described above.
    """

    dists_x_to_y, corrs_x_to_y = pcu.k_nearest_neighbors(
        x, y, k=1, squared_distances=False, max_points_per_leaf=max_points_per_leaf
    )
    dists_y_to_x, corrs_y_to_x = pcu.k_nearest_neighbors(
        y, x, k=1, squared_distances=False, max_points_per_leaf=max_points_per_leaf
    )

    dists_x_to_y = np.linalg.norm(x[corrs_y_to_x] - y, axis=-1, ord=p_norm).mean()
    dists_y_to_x = np.linalg.norm(y[corrs_x_to_y] - x, axis=-1, ord=p_norm).mean()

    Comp = np.mean(dists_x_to_y)
    Acc = np.mean(dists_y_to_x)
    cham_dist = Comp + Acc

    if return_index:
        return cham_dist, corrs_x_to_y, corrs_y_to_x

    return cham_dist, Acc, Comp


def compute_chamfer_distance(pred_sampled, gt_points):
    """
    Compute chamfer distance between predicted and ground truth points.

    Args:
        pred_sampled (numpy.ndarray): Predicted point cloud.
        gt_points (numpy.ndarray): Ground truth points.

    Returns:
        float: Chamfer distance.
    """
    chamfer_dist, acc, comp = chamfer_distance(pred_sampled, gt_points)
    return chamfer_dist, acc, comp


def f_score(precision, recall):
    """
    Compute F-score.

    Args:
        precision (float): Precision.
        recall (float): Recall.

    Returns:
        float: F-score.
    """
    return 2 * precision * recall / (precision + recall)


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


def compute_precision_recall_IOU(
    pred_sampled, gt_points, metrics, thresh_list=[0.02], edge_type="all"
):
    """
    Compute precision, recall, F-score, and IOU.

    Args:
        pred_sampled (numpy.ndarray): Predicted point cloud.
        gt_points (numpy.ndarray): Ground truth points.
        metrics (dict): Dictionary to store metrics.

    Returns:
        dict: Updated metrics.
    """
    if edge_type == "all":
        for thresh in thresh_list:
            dists_a_to_b, _ = pcu.k_nearest_neighbors(
                pred_sampled, gt_points, k=1
            )  # k closest points (in pts_b) for each point in pts_a
            correct_pred = np.sum(dists_a_to_b < thresh)
            precision = correct_pred / len(dists_a_to_b)
            metrics[f"precision_{thresh}"].append(precision)

            dists_b_to_a, _ = pcu.k_nearest_neighbors(gt_points, pred_sampled, k=1)
            correct_gt = np.sum(dists_b_to_a < thresh)
            recall = correct_gt / len(dists_b_to_a)
            metrics[f"recall_{thresh}"].append(recall)

            fscore = 2 * precision * recall / (precision + recall)
            metrics[f"fscore_{thresh}"].append(fscore)

            intersection = min(correct_pred, correct_gt)
            union = (
                len(dists_a_to_b) + len(dists_b_to_a) - max(correct_pred, correct_gt)
            )

            IOU = intersection / union
            metrics[f"IOU_{thresh}"].append(IOU)
        return metrics
    else:
        correct_gt_list = []
        correct_pred_list = []
        _, acc, comp = compute_chamfer_distance(pred_sampled, gt_points)
        for thresh in thresh_list:
            dists_b_to_a, _ = pcu.k_nearest_neighbors(gt_points, pred_sampled, k=1)
            correct_gt = np.sum(dists_b_to_a < thresh)
            num_gt = len(dists_b_to_a)
            correct_gt_list.append(correct_gt)
            dists_a_to_b, _ = pcu.k_nearest_neighbors(pred_sampled, gt_points, k=1)
            correct_pred = np.sum(dists_a_to_b < thresh)
            correct_pred_list.append(correct_pred)
            num_pred = len(dists_a_to_b)

        return correct_gt_list, num_gt, correct_pred_list, num_pred, acc, comp


def get_gt_points(
    scan_name,
    edge_type="all",
    interval=0.005,
    return_direction=False,
    data_base_dir=None,
):
    """
    Get ground truth points from a dataset.

    Args:
        name (str): Name of the dataset.

    Returns:
        numpy.ndarray: Raw and processed ground truth points.
    """
    objs_dir = os.path.join(data_base_dir, "obj")
    obj_names = os.listdir(objs_dir)
    obj_names.sort()
    index_obj_names = {}
    for obj_name in obj_names:
        index_obj_names[obj_name[:8]] = obj_name

    json_feats_path = os.path.join(data_base_dir, "chunk_0000_feats.json")
    with open(json_feats_path, "r") as f:
        json_data_feats = json.load(f)
    json_stats_path = os.path.join(data_base_dir, "chunk_0000_stats.json")
    with open(json_stats_path, "r") as f:
        json_data_stats = json.load(f)

    # get the normalize scale to help align the nerf points and gt points
    [
        x_min,
        y_min,
        z_min,
        x_max,
        y_max,
        z_max,
        x_range,
        y_range,
        z_range,
    ] = json_data_stats[scan_name]["bbox"]
    scale = 1 / max(x_range, y_range, z_range)
    poi_center = (
        np.array([((x_min + x_max) / 2), ((y_min + y_max) / 2), ((z_min + z_max) / 2)])
        * scale
    )
    set_location = [0.5, 0.5, 0.5] - poi_center  # based on the rendering settings
    obj_path = os.path.join(objs_dir, index_obj_names[scan_name])

    with open(obj_path, encoding="utf-8") as file:
        data = file.readlines()

    vertices_obj = [each.split(" ") for each in data if each.split(" ")[0] == "v"]
    vertices_xyz = [
        [float(v[1]), float(v[2]), float(v[3].replace("\n", ""))] for v in vertices_obj
    ]

    edge_pts = []
    edge_pts_raw = []
    edge_pts_direction = []
    rename = {
        "BSpline": "curve",
        "Circle": "curve",
        "Ellipse": "curve",
        "Line": "line",
    }
    for each_curve in json_data_feats[scan_name]:
        if edge_type != "all" and rename[each_curve["type"]] != edge_type:
            continue

        if each_curve["sharp"]:  # each_curve["type"]: BSpline, Line, Circle
            each_edge_pts = [vertices_xyz[i] for i in each_curve["vert_indices"]]
            edge_pts_raw += each_edge_pts

            gt_sampling = []
            each_edge_pts = np.array(each_edge_pts)
            for index in range(len(each_edge_pts) - 1):
                next = each_edge_pts[index + 1]
                current = each_edge_pts[index]
                num = int(np.linalg.norm(next - current) // interval)
                linspace = np.linspace(0, 1, num)
                gt_sampling.append(
                    linspace[:, None] * current + (1 - linspace)[:, None] * next
                )

                if return_direction:
                    direction = (next - current) / np.linalg.norm(next - current)
                    edge_pts_direction.extend([direction] * num)
            each_edge_pts = np.concatenate(gt_sampling).tolist()
            edge_pts += each_edge_pts

    if len(edge_pts_raw) == 0:
        return None, None, None

    edge_pts_raw = np.array(edge_pts_raw) * scale + set_location
    edge_pts = np.array(edge_pts) * scale + set_location
    edge_pts_direction = np.array(edge_pts_direction)

    return (
        edge_pts_raw.astype(np.float32),
        edge_pts.astype(np.float32),
        edge_pts_direction,
    )


def get_pred_points_and_directions(
    json_path,
    sample_resolution=0.005,
):
    with open(json_path, "r") as f:
        json_data = json.load(f)

    curve_paras = np.array(json_data["curves_ctl_pts"]).reshape(-1, 3)
    curves_ctl_pts = curve_paras.reshape(-1, 4, 3)
    lines_end_pts = np.array(json_data["lines_end_pts"]).reshape(-1, 2, 3)

    num_curves = len(curves_ctl_pts)
    num_lines = len(lines_end_pts)

    all_curve_points = []
    all_curve_directions = []

    # # -----------------------------------for Cubic Bezier-----------------------------------
    if num_curves > 0:
        for i, each_curve in enumerate(curves_ctl_pts):
            each_curve = np.array(each_curve).reshape(4, 3)  # shape: (4, 3)
            # sample_num = int(np.linalg.norm(each_curve[0] - each_curve[-1]) // 0.01)
            sample_num = int(
                bezier_curve_length(each_curve, num_samples=100) // sample_resolution
            )
            t = np.linspace(0, 1, sample_num)
            matrix_u = np.array([t**3, t**2, t, [1] * sample_num]).reshape(
                4, sample_num
            )

            matrix_middle = np.array(
                [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
            )

            matrix = np.matmul(
                np.matmul(matrix_u.T, matrix_middle), each_curve
            ).reshape(sample_num, 3)

            # curve_points_flatten = matrix.reshape(-1)
            all_curve_points += matrix.tolist()

            # Calculate the curve directions (derivatives)
            derivative_u = 3 * t**2
            derivative_v = 2 * t
            # derivative_constant = np.ones_like(t)

            # Derivative matrices for x, y, z
            dx = (
                (
                    -3 * each_curve[0][0]
                    + 9 * each_curve[1][0]
                    - 9 * each_curve[2][0]
                    + 3 * each_curve[3][0]
                )
                * derivative_u
                + (6 * each_curve[0][0] - 12 * each_curve[1][0] + 6 * each_curve[2][0])
                * derivative_v
                + (-3 * each_curve[0][0] + 3 * each_curve[1][0])
            )

            dy = (
                (
                    -3 * each_curve[0][1]
                    + 9 * each_curve[1][1]
                    - 9 * each_curve[2][1]
                    + 3 * each_curve[3][1]
                )
                * derivative_u
                + (6 * each_curve[0][1] - 12 * each_curve[1][1] + 6 * each_curve[2][1])
                * derivative_v
                + (-3 * each_curve[0][1] + 3 * each_curve[1][1])
            )

            dz = (
                (
                    -3 * each_curve[0][2]
                    + 9 * each_curve[1][2]
                    - 9 * each_curve[2][2]
                    + 3 * each_curve[3][2]
                )
                * derivative_u
                + (6 * each_curve[0][2] - 12 * each_curve[1][2] + 6 * each_curve[2][2])
                * derivative_v
                + (-3 * each_curve[0][2] + 3 * each_curve[1][2])
            )
            for i in range(sample_num):
                direction = np.array([dx[i], dy[i], dz[i]])
                norm_direction = direction / np.linalg.norm(direction)
                all_curve_directions.append(norm_direction)

    all_line_points = []
    all_line_directions = []
    # # -------------------------------------for Line-----------------------------------------
    if num_lines > 0:
        for i, each_line in enumerate(lines_end_pts):
            each_line = np.array(each_line).reshape(2, 3)  # shape: (2, 3)
            sample_num = int(
                np.linalg.norm(each_line[0] - each_line[-1]) // sample_resolution
            )
            t = np.linspace(0, 1, sample_num)

            matrix_u_l = np.array([t, [1] * sample_num])
            matrix_middle_l = np.array([[-1, 1], [1, 0]])

            matrix_l = np.matmul(
                np.matmul(matrix_u_l.T, matrix_middle_l), each_line
            ).reshape(sample_num, 3)
            # line_points = matrix_l.transpose(0, 1)
            # line_points_flatten = matrix_l.reshape(-1)
            all_line_points += matrix_l.tolist()

            # Calculate the direction vector for the line segment
            direction = each_line[1] - each_line[0]
            norm_direction = direction / np.linalg.norm(direction)

            for point in matrix_l:
                all_line_directions.append(norm_direction)

    all_curve_points = np.array(all_curve_points).reshape(-1, 3)
    all_line_points = np.array(all_line_points).reshape(-1, 3)
    return all_curve_points, all_line_points, all_curve_directions, all_line_directions


def downsample_point_cloud_average(
    points, num_voxels_per_axis=256, min_bound=None, max_bound=None
):
    """
    Downsample a point set based on the number of voxels per axis by averaging the points within each voxel.

    Args:
        points: a [#v, 3]-shaped array of 3d points.
        num_voxels_per_axis: a scalar or 3-tuple specifying the number of voxels along each axis.

    Returns:
        A [#v', 3]-shaped numpy array of downsampled points, where #v' is the number of occupied voxels.
    """

    # Calculate the bounding box of the point cloud
    if min_bound is None:
        min_bound = np.min(points, axis=0)
    else:
        min_bound = np.array(min_bound)

    if max_bound is None:
        max_bound = np.max(points, axis=0)
    else:
        max_bound = np.array(max_bound)

    # Determine the size of the voxel based on the desired number of voxels per axis
    if isinstance(num_voxels_per_axis, int):
        voxel_size = (max_bound - min_bound) / num_voxels_per_axis
    else:
        voxel_size = [
            (max_bound[i] - min_bound[i]) / num_voxels_per_axis[i] for i in range(3)
        ]

    # Use the existing function to downsample the point cloud based on voxel size
    downsampled_points = pcu.downsample_point_cloud_on_voxel_grid(
        voxel_size, points, min_bound=min_bound, max_bound=max_bound
    )

    return downsampled_points
