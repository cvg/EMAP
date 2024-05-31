from src.edge_extraction.edge_fitting.main import save_3d_lines_to_file
from src.edge_extraction.edge_fitting.line_fit import line_fitting
from src.edge_extraction.edge_fitting.bezier_fit import (
    bezier_fit,
    bezier_curve,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.sparse.csgraph import connected_components
import open3d as o3d
from scipy.spatial.distance import euclidean, cdist
import os


def line_segment_point_distance(line_segment, query_point):
    """Compute the Euclidean distance between a line segment and a query point.

    Parameters:
        line_segment (np.ndarray): An array of shape (6,), representing two 3D endpoints.
        query_point (np.ndarray): An array of shape (3,), representing the 3D query point.

    Returns:
        float: The minimum distance from the query point to the line segment.
    """
    point1, point2 = line_segment[:3], line_segment[3:]
    point_delta = point2 - point1
    u = np.clip(
        np.dot(query_point - point1, point_delta) / np.dot(point_delta, point_delta),
        0,
        1,
    )
    closest_point = point1 + u * point_delta
    return np.linalg.norm(closest_point - query_point)


def compute_pairwise_distances(line_segments):
    """Compute pairwise distances between line segments.

    Parameters:
        line_segments (np.ndarray): An array of shape (N, 6), each row represents a line segment in 3D.

    Returns:
        np.ndarray: A symmetric array of shape (N, N), containing pairwise distances.
    """
    num_lines = len(line_segments)
    endpoints = line_segments.reshape(-1, 3)
    dist_matrix = np.zeros((num_lines, num_lines))

    for i, line_segment in enumerate(line_segments):
        for j in range(i + 1, num_lines):
            min_distance = min(
                line_segment_point_distance(line_segment, endpoints[2 * j]),
                line_segment_point_distance(line_segment, endpoints[2 * j + 1]),
            )
            dist_matrix[i, j] = min_distance

    dist_matrix += dist_matrix.T  # Make the matrix symmetric
    return dist_matrix


def compute_pairwise_cosine_similarity(line_segments):
    direction_vectors = line_segments[:, 3:] - line_segments[:, :3]
    pairwise_similarity = cosine_similarity(direction_vectors)
    return pairwise_similarity


def bezier_curve_distance(points1, points2):
    distances = np.linalg.norm(points1[:, np.newaxis] - points2, axis=2)
    min_distance = np.min(distances)
    return min_distance


def bezier_slope_vector(P0, P1, P2, P3, t):
    # Calculate the derivative of the 3D Bézier curve
    dp_dt = (
        -3 * (1 - t) ** 2 * P0
        + 3 * (1 - 4 * t + 3 * t**2) * P1
        + 3 * (2 * t - 3 * t**2) * P2
        + 3 * t**2 * P3
    )
    return dp_dt


def get_dist_similarity(control_points1, control_points2, points1, points2, t_values):
    control_points1 = control_points1.reshape(-1, 3)
    control_points2 = control_points2.reshape(-1, 3)

    # Calculate the pairwise distances between all points in the two sets
    distances = cdist(points1, points2)

    # Find the indices of the minimum distance pair
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_indices]

    points_slope1 = bezier_slope_vector(
        control_points1[0],
        control_points1[1],
        control_points1[2],
        control_points1[3],
        t_values[min_indices[0]],
    )

    points_slope2 = bezier_slope_vector(
        control_points2[0],
        control_points2[1],
        control_points2[2],
        control_points2[3],
        t_values[min_indices[1]],
    )

    # Compute cosine similarity between the two slopes
    similarity = np.abs(np.dot(points_slope1, points_slope2)) / (
        np.linalg.norm(points_slope1) * np.linalg.norm(points_slope2)
    )

    return min_distance, similarity


def merge_line_segments(
    line_segments, raw_points_on_lines, distance_threshold, similarity_threshold
):
    dist_matrix = compute_pairwise_distances(line_segments)
    similarity_matrix = compute_pairwise_cosine_similarity(line_segments)

    # Create adjacency matrix based on distance and similarity thresholds
    adjacency_matrix = (dist_matrix <= distance_threshold) & (
        similarity_matrix >= similarity_threshold
    )
    # Compute connected components
    num_components, labels = connected_components(adjacency_matrix)

    merged_line_segments = []
    for component in range(num_components):
        component_indices = np.where(labels == component)[0]
        if len(component_indices) == 1:
            merged_line_segments.append(line_segments[component_indices[0]])
            continue
        else:
            raw_points_on_lines_group = raw_points_on_lines[component_indices]
            raw_points_on_lines_group_array = np.array(
                [
                    point
                    for raw_points_on_lines in raw_points_on_lines_group
                    for point in raw_points_on_lines
                ]
            ).reshape(-1, 3)

            try:
                line_segment, _ = line_fitting(raw_points_on_lines_group_array)
                merged_line_segments.append(line_segment)
            except:
                continue

    merged_line_segments = np.array(merged_line_segments)
    return merged_line_segments


def merge_bezier_curves(
    control_points_list,
    raw_points_on_curves,
    distance_threshold,
    similarity_threshold,
    num_samples=100,
):
    """
    Merge Bezier curves based on distance and similarity thresholds.

    Parameters:
        control_points_list (array): list of control points for each curve.
        raw_points_on_curves (array): actual points on each Bezier curve.
        distance_threshold (float): maximum distance for merging curves.
        similarity_threshold (float): minimum similarity for merging curves.
        num_samples (int, optional): number of samples for curve points.

    Returns:
        np.array: Merged control points for the Bezier curves.
    """

    if not isinstance(control_points_list, np.ndarray) or not isinstance(
        raw_points_on_curves, np.ndarray
    ):
        raise ValueError("Input must be NumPy arrays.")

    num_curves = len(control_points_list)
    dist_matrix = np.zeros((num_curves, num_curves))
    similarity_matrix = np.zeros((num_curves, num_curves))
    t_values = np.linspace(0, 1, num_samples)

    for i, control_points1 in enumerate(control_points_list):
        for j in range(i + 1, num_curves):
            control_points2 = control_points_list[j]
            points1 = bezier_curve(t_values, *(control_points1.tolist())).reshape(-1, 3)
            points2 = bezier_curve(t_values, *(control_points2.tolist())).reshape(-1, 3)
            dist_matrix[i, j], similarity_matrix[i, j] = get_dist_similarity(
                control_points1, control_points2, points1, points2, t_values
            )

    dist_matrix += dist_matrix.T
    similarity_matrix += similarity_matrix.T

    adjacency_matrix = (dist_matrix <= distance_threshold) & (
        similarity_matrix >= similarity_threshold
    )
    num_components, labels = connected_components(adjacency_matrix)

    merged_bezier_curves = []
    for component in range(num_components):
        component_indices = np.where(labels == component)[0]
        if len(component_indices) == 1:
            p = control_points_list[component_indices[0]]
            merged_bezier_curves.append(p)
        else:
            raw_points_group = [raw_points_on_curves[i] for i in component_indices]
            raw_points_group_array = np.concatenate(raw_points_group, axis=0)
            p = bezier_fit(raw_points_group_array)
            merged_bezier_curves.append(p)

    return np.array(merged_bezier_curves)


def merge_endpoints(merged_line_segments, merged_bezier_curves, distance_threshold):
    N_lines = len(merged_line_segments)
    N_curves = len(merged_bezier_curves)

    if N_lines == 0 and N_curves == 0:
        return [], []

    if N_lines > 0:
        line_endpoints = merged_line_segments.reshape(-1, 3)
    else:
        line_endpoints = np.array([]).reshape(-1, 3)

    if N_curves > 0:
        curve_endpoints = merged_bezier_curves[:, [0, 1, 2, -3, -2, -1]].reshape(-1, 3)
    else:
        curve_endpoints = np.array([]).reshape(-1, 3)

    concat_endpoints = np.concatenate([line_endpoints, curve_endpoints], axis=0)

    dist_matrix = cdist(concat_endpoints, concat_endpoints)
    adjacency_matrix = dist_matrix <= distance_threshold
    num_components, labels = connected_components(adjacency_matrix)
    for component in range(num_components):
        component_indices = np.where(labels == component)[0]
        if len(component_indices) > 1:
            endpoints = concat_endpoints[component_indices]
            mean_endpoint = np.mean(endpoints, axis=0)
            concat_endpoints[component_indices] = mean_endpoint

    if N_lines > 0:
        merged_line_segments_merged_endpoints = concat_endpoints[: N_lines * 2].reshape(
            -1, 6
        )
    else:
        merged_line_segments_merged_endpoints = []

    if N_curves > 0:
        merged_curve_segments_merged_endpoints = np.zeros_like(merged_bezier_curves)
        curve_merged_endpoints = concat_endpoints[N_lines * 2 :].reshape(-1, 6)
        merged_curve_segments_merged_endpoints[:, :3] = curve_merged_endpoints[:, :3]
        merged_curve_segments_merged_endpoints[:, 3:9] = merged_bezier_curves[:, 3:9]
        merged_curve_segments_merged_endpoints[:, 9:] = curve_merged_endpoints[:, 3:]

    else:
        merged_curve_segments_merged_endpoints = []

    return merged_line_segments_merged_endpoints, merged_curve_segments_merged_endpoints


def approximate_curve_length(P0, P1, P2, P3):
    return np.linalg.norm(P1 - P0) + np.linalg.norm(P2 - P1) + np.linalg.norm(P3 - P2)


def generate_points_curve(curves):
    all_points = []
    for curve in curves:
        P0 = np.array(curve[:3])
        P1 = np.array(curve[3:6])
        P2 = np.array(curve[6:9])
        P3 = np.array(curve[9:])

        # Approximate number of points based on curve length
        length = approximate_curve_length(P0, P1, P2, P3)
        num_points = int(np.ceil(length * 1000))  # Adjust the factor as needed

        t_values = np.linspace(0, 1, num_points)
        curve_points = np.array(bezier_curve(t_values, *(curve.tolist())))

        all_points.extend(curve_points)

    return np.array(all_points).reshape(-1, 3)


def merge(
    out_dir,
    fitted_edge_dict,
    merge_edge_distance_threshold=5.0,
    merge_endpoints_distance_threshold=1.0,
    merge_similarity_threshold=0.98,
    merge_endpoints_flag=True,
    merge_edge_flag=True,
    merge_curve_flag=False,
    save_ply=False,
):

    resolution = int(fitted_edge_dict["resolution"])
    lines = np.array(fitted_edge_dict["lines_end_pts"]).reshape(-1, 6)
    raw_points_on_lines = np.array(
        fitted_edge_dict["raw_points_on_lines"], dtype=object
    )
    bezier_curves = np.array(fitted_edge_dict["curves_ctl_pts"]).reshape(-1, 12)
    raw_points_on_curves = np.array(
        fitted_edge_dict["raw_points_on_curves"], dtype=object
    )

    # Normalize thresholds
    merge_edge_distance_threshold /= resolution
    merge_endpoints_distance_threshold /= resolution

    # Merge lines
    if merge_edge_flag and len(lines) > 0:
        merged_line_segments = merge_line_segments(
            lines,
            raw_points_on_lines,
            merge_edge_distance_threshold / 2.0,
            merge_similarity_threshold,
        )
    else:
        merged_line_segments = lines

    if merge_curve_flag and merge_edge_flag:
        if len(bezier_curves) > 0:
            merged_bezier_curves = merge_bezier_curves(
                bezier_curves,
                raw_points_on_curves,
                merge_edge_distance_threshold,
                merge_similarity_threshold,
            )
        else:
            merged_bezier_curves = []
    else:
        merged_bezier_curves = bezier_curves

    if merge_endpoints_flag:
        (
            merged_line_segments,
            merged_bezier_curves,
        ) = merge_endpoints(
            merged_line_segments,
            merged_bezier_curves,
            merge_endpoints_distance_threshold,
        )

    if save_ply:
        if len(merged_line_segments) > 0:
            save_3d_lines_to_file(
                merged_line_segments,
                os.path.join(out_dir, "merged_line_segments.ply"),
                width=2,
                scale=1.0,
            )
            print(f"Saved merged line segments to {out_dir}.")

        if len(merged_bezier_curves) > 0:
            pcd = o3d.geometry.PointCloud()
            meregd_bezier_curves_points = generate_points_curve(merged_bezier_curves)
            pcd.points = o3d.utility.Vector3dVector(meregd_bezier_curves_points)
            o3d.io.write_point_cloud(
                os.path.join(out_dir, "merged_bezier_curve_points.ply"),
                pcd,
                write_ascii=True,
            )
            print(f"Saved merged bezier curves to {out_dir}.")

    merged_edge_dict = {
        "lines_end_pts": (
            merged_line_segments.tolist() if len(merged_line_segments) > 0 else []
        ),
        "curves_ctl_pts": (
            merged_bezier_curves.tolist() if len(merged_bezier_curves) > 0 else []
        ),
    }

    return merged_edge_dict
