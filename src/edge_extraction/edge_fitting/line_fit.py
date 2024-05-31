import numpy as np


def split_into_monotonic_sublists(numbers, max_longsublists=2, min_length=4):
    if not numbers:
        return []

    # Initialize list to store continuous and monotonic sublists
    monotonic_sublists = []
    current_sublist = [numbers[0]]

    # Create continuous and monotonic sublists from the original list
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_sublist.append(numbers[i])
        else:
            if len(current_sublist) > 1:
                monotonic_sublists.append(tuple(current_sublist))
            current_sublist = [numbers[i]]

    # Add the last continuous and monotonic sublist if it contains more than one element
    if len(current_sublist) > 1:
        monotonic_sublists.append(tuple(current_sublist))

    # Convert to set and back to list to remove duplicates
    monotonic_sublists = list(set(monotonic_sublists))
    # Sort the sublists by length in descending order
    monotonic_sublists.sort(key=len, reverse=True)

    # Keep the specified number of longest sublists
    max_sublists = min(max_longsublists, len(monotonic_sublists))
    long_sublists = monotonic_sublists[:max_sublists]
    short_sublists = monotonic_sublists[max_sublists:]

    # Handle sublists that are too short
    sublists_out_curves = []
    for sublist in long_sublists:
        if len(sublist) < min_length:
            short_sublists.append(sublist)
        else:
            sublists_out_curves.append(list(sublist))

    # Split the remaining short sublists into pairs of numbers
    sublists_out_lines = []
    for sublist in short_sublists:
        for j in range(len(sublist) - 1):
            sublists_out_lines.append([sublist[j], sublist[j + 1]])

    return [list(t) for t in sublists_out_curves], [list(t) for t in sublists_out_lines]


def fit_line_ransac_3d(
    points_wld,
    voxel_size=256,
    max_iterations=100,
    min_inliers=4,
    max_lines=3,
    max_curves=2,
    keep_short_lines=False,
    ransac_with_direction=False,
):
    """
    Fit multiple lines to 3D points using RANSAC.

    Parameters:
    - points (numpy.ndarray): Array of 3D points.
    - voxel_size (float): Voxel size for inlier distance threshold.
    - max_iterations (int): Maximum number of RANSAC iterations.
    - min_inliers (int): Minimum number of inliers required to consider a line.
    - max_lines (int): Maximum number of lines to fit.

    Returns:
    - best_endpoints (list): List of line endpoints (start and end points).
    - split_points (list): List of points belonging to each fitted line.
    - remaining_points (numpy.ndarray): Points not assigned to any line.
    - remaining_point_indices (list): Indices of remaining points in the original input points.
    """
    inlier_dist_threshold = 1.0 / voxel_size  # 1.0 / voxel_size
    best_lines = []
    best_endpoints = []
    split_points = []
    # remaining_point_indices = []  # List to store indices of remaining points
    N_points = len(points_wld)
    remaining_point_indices = np.arange(N_points)
    min_inlier_ratio = 1.0 / max_lines
    raw_points_wld = points_wld.copy()

    while max_lines and len(points_wld) >= min_inliers:
        max_lines -= 1
        best_line = None
        best_inliers_mask = None
        best_num_inliers = 0

        if not ransac_with_direction:
            for _ in range(max_iterations):
                # Generate all unique combinations of point pairs
                sample_indices = np.random.choice(len(points_wld), 2, replace=False)
                sample_points_wld = points_wld[sample_indices, :3]

                p1, p2 = sample_points_wld
                direction = p2 - p1

                if np.linalg.norm(direction) < 1e-6:
                    continue

                direction /= np.linalg.norm(direction)

                distances = np.linalg.norm(
                    np.cross(points_wld[:, :3] - p1, direction), axis=1
                )

                inliers_mask = distances < inlier_dist_threshold
                num_inliers = np.sum(inliers_mask)

                if num_inliers > best_num_inliers:
                    best_line = (p1, direction)
                    best_num_inliers = num_inliers
                    best_inliers_mask = inliers_mask

        else:
            points = points_wld[:, :3]
            direction = points_wld[:, 3:]
            normalized_direction = direction / np.linalg.norm(
                direction, axis=1, keepdims=True
            )

            distance = np.linalg.norm(
                np.cross(points - points[:, None], normalized_direction), axis=2
            )  # N x N
            inlier_mask = distance < inlier_dist_threshold
            num_inliers = np.sum(inlier_mask, axis=1)

            best_inlier_idx = np.argmax(num_inliers)
            best_line = (points[best_inlier_idx], direction[best_inlier_idx])
            best_inliers_mask = inlier_mask[best_inlier_idx]
            best_num_inliers = num_inliers[best_inlier_idx]

        if best_num_inliers >= min_inliers:
            p1, direction = best_line
            inlier_points = points_wld[best_inliers_mask, :3]
            inlier_ratio_pred = best_num_inliers / N_points
            if inlier_ratio_pred < min_inlier_ratio:
                break

            center = np.mean(inlier_points, axis=0)
            endpoints_centered = inlier_points - center
            u, s, vh = np.linalg.svd(endpoints_centered, full_matrices=False)
            updated_direction = vh[0]
            updated_direction = updated_direction / np.linalg.norm(updated_direction)

            projections = np.dot(inlier_points - p1, updated_direction)
            line_segment = np.zeros(6)
            line_segment[:3] = p1 + np.min(projections) * updated_direction
            line_segment[3:] = p1 + np.max(projections) * updated_direction

            points_wld = points_wld[~best_inliers_mask]
            split_points.append(inlier_points.tolist())
            remaining_point_indices = remaining_point_indices[~best_inliers_mask]

            best_lines.append(best_line)
            best_endpoints.append(line_segment)

    # find potential curve points
    if len(remaining_point_indices) > 0:
        potential_curve_indices, shortline_indices = split_into_monotonic_sublists(
            remaining_point_indices.tolist(), max_curves
        )
        potential_curve_points = [
            raw_points_wld[potential_curve_indice, :3]
            for potential_curve_indice in potential_curve_indices
        ]
        if keep_short_lines and len(shortline_indices) > 0:
            shortline_points = raw_points_wld[shortline_indices, :3]
            shortline_points = shortline_points.reshape(-1, 6)
            best_endpoints.extend(shortline_points)
            split_points.extend(shortline_points.reshape(-1, 2, 3).tolist())
    else:
        potential_curve_points = []

    return best_endpoints, split_points, potential_curve_points


def line_fitting(endpoints):
    center = np.mean(endpoints, axis=0)

    # compute the main direction through SVD
    endpoints_centered = endpoints - center
    u, s, vh = np.linalg.svd(endpoints_centered, full_matrices=False)
    lamda = s[0] / np.sum(s)
    main_direction = vh[0]
    main_direction = main_direction / np.linalg.norm(main_direction)

    # project endpoints onto the main direction
    projections = []
    for endpoint_centered in endpoints_centered:
        projections.append(np.dot(endpoint_centered, main_direction))
    projections = np.array(projections)

    # construct final line
    straight_line = np.zeros(6)
    # print(np.min(projections), np.max(projections))
    straight_line[:3] = center + main_direction * np.min(projections)
    straight_line[3:] = center + main_direction * np.max(projections)

    return straight_line, lamda


def lines_fitting(lines, lamda_threshold):
    straight_lines = []
    curve_line_segments_candidate = []
    curves = []
    lamda_list = []
    for endpoints in lines:
        # merge line segments into a final line segment
        # total least squares on endpoints
        straight_line, lamda = line_fitting(endpoints)
        lamda_list.append(lamda)
        if lamda < lamda_threshold:
            curves.append(endpoints)
            curve_line_segments_candidate.append(
                [
                    np.hstack([endpoints[i], endpoints[i + 1]])
                    for i in range(len(endpoints) - 1)
                ]
            )
            continue

        straight_lines.append(straight_line)
    straight_lines = np.array(straight_lines)
    curve_line_segments_candidate = np.array(curve_line_segments_candidate)
    return (
        straight_lines,
        curve_line_segments_candidate,
        curves,
        lamda_list,
    )
